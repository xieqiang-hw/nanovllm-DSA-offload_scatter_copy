# kvcache_scatter_copy

一个 caller-owned 的 DRAM→HBM scatter 算子，按输入 dtype 选择 BF16 或 packed-C8；同一份源码在本机针对 A3/A5 编译。

| 平台 | BF16：512 CKV＋64 RoPE 元素，1152 B/token | packed-C8：656 B/token |
|---|---|---|
| A3 / Ascend 910C | 支持目标，待硬件验收 | 支持目标，待硬件验收 |
| A5 / Ascend 950 | 支持目标，待硬件验收 | 支持目标，待硬件验收 |

## 构建

先加载本机匹配的 CANN 开发环境，并安装对应的 `torch`、`torch_npu`；需要 `msopgen` 和 `torch_npu.empty_with_swapped_memory`。

```bash
bash build.sh
export PYTHONPATH="$PWD/torch_extension${PYTHONPATH:+:$PYTHONPATH}"
```

构建自动识别 SoC，一次包含 BF16/C8，产物保存在 `build/<soc>/`。显式指定目标可用 `SOC_VERSION=ascend910_93 bash build.sh` 或 `SOC_VERSION=ascend950 bash build.sh`；不同架构的机器分别编译。Python 包只加载当前仓库中匹配本机 SoC 的 OPP。`PYTHON`、`MAX_JOBS` 仅配置构建解释器和并行度。

## 接口

```python
import kvcache_ops  # 在首次 NPU 操作之前导入，注册本仓库算子
import torch

# torch.ops.kvcache_ops.kvcache_scatter_copy 的签名
# Optional 表示可传 None；九个参数都必须显式传入。
def kvcache_scatter_copy(
    hbm_kv: torch.Tensor,             # caller-owned，原地更新
    dram_kv: torch.Tensor,            # 只读
    hbm_kpe: torch.Tensor | None,     # BF16 必填，原地更新；C8 传 None
    dram_kpe: torch.Tensor | None,    # BF16 必填，只读；C8 传 None
    hbm_block_table: torch.Tensor,    # int32[B, HBM_MAX_BLOCKS]
    dram_block_table: torch.Tensor,   # int32[B, DRAM_MAX_BLOCKS]
    source_token_ids: torch.Tensor,   # int32[B, 1, C]
    destination_slots: torch.Tensor,  # int32[B, 1, C]
    copy_counts: torch.Tensor,        # int32[B]
) -> None: ...
```

| 参数 | BF16 模式 | packed-C8 模式 |
|---|---|---|
| `hbm_kv` | `bfloat16[H,128,1,512]` | `int8[H,128,1,656]` |
| `dram_kv` | `bfloat16[D,128,1,512]` | `int8[D,128,1,656]` |
| `hbm_kpe` | `bfloat16[H,128,1,64]` | `None` |
| `dram_kpe` | `bfloat16[D,128,1,64]` | `None` |

`H/D` 为 HBM/DRAM 物理 block 数。模式由 `hbm_kv.dtype` 决定，源/目标 dtype 必须一致。所有张量为同一 NPU 上的连续基础格式，DRAM 源使用设备可访问的主机内存。BF16 保持分离存储；旧三维缓存可提前 `unsqueeze(2)` 得到零拷贝视图。C8 传 packed cache 的 `int8` 字节视图，原样搬移整个 656-byte row，不进行量化、反量化或格式转换。

`1 <= C <= 65536`，每请求只处理前 `copy_counts[b]` 项，数量必须在 `[0,C]`；0 严格不修改该请求的缓存，未使用的后缀不读取。逻辑 token/slot 经对应 block table 映射到物理行。caller 保证有效索引和物理 block 合法、源目标不重叠、有效目标物理行全局唯一；允许重复源读取。算子返回 `None`，只修改命中的 HBM 行，源与 metadata 不变，不分配输出缓存，也不做 CPU 同步。

```python
# BF16
kvcache_ops.kvcache_scatter_copy(
    hbm_ckv, dram_ckv, hbm_kpe, dram_kpe,
    hbm_block_table, dram_block_table, src_ids, dst_slots, copy_counts,
)
# packed-C8
kvcache_ops.kvcache_scatter_copy(
    hbm_kv_bytes, dram_kv_bytes, None, None,
    hbm_block_table, dram_block_table, src_ids, dst_slots, copy_counts,
)
```

## 单 die 验证

```bash
python3 tests/test_scatter_copy.py --device npu:0 --dtype bf16 c8 --batch-size 2 --copy-count 100
# 增加图捕获及修改 metadata 后的重放验证
python3 tests/test_scatter_copy.py --device npu:0 --dtype bf16 c8 --batch-size 2 --copy-count 100 --graph
```

逐字节检查复制内容、源/metadata 只读、caller-owned 地址、零 count、不同请求不同 count、跨 block、重复源、未使用后缀、guard block、完整容量及非默认 stream；C8 包含相邻行的非对齐搬运。失败非零退出。

## 多 die 带宽测试

测试参数只通过命令行传递，BF16/C8 顺序执行。`--cards` 按逻辑设备/die 计数：A3 的 8 张物理卡、16 die 可指定到 16。从 `--visible-devices` 列表依次取前 N 个 die，每个 die 一个进程；未指定列表时沿用运行时可见设备。`--batch-size` 是每 die 的请求数，`--copy-count` 是每请求固定搬移的 token 数。

```bash
bash tests/run_scatter_copy_multiprocess.sh \
  --visible-devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --cards 1 2 4 8 16 --batch-size 8 16 24 32 --copy-count 0 100 200 300 500 2048
```

默认 `cards=1, batch-size=8, copy-count=300`；可用 `--source-len 65536 --hbm-slots 8192 --copy-cap 16384 --warmup 10 --iters 1000 --seed 7 --timeout 600` 调整其余参数（此处即默认值）。copy-count 不得超过 copy-cap、source-len、hbm-slots。`--dry-run` 仅校验并列出用例，无需 Torch/CANN。

各请求使用互不重叠的 DRAM 源池；两种 dtype 的同一用例使用一致的索引和 block table。初始化、正确性检查、预热不计时；所有 die 同步后，用 NPU Event 测量重复调用的平均耗时。所有 die 完成计时后才开始结果验证。失败/超时终止进程并非零退出。

只打印运行进度和最后一张表，不生成 log、CSV、JSON、临时同步文件或其它报告：

```text
data-type  card-count  batch-size  copy-count  avg_us_mean  avg_us_max  avg_bandwidth
```

令每个 die 的单次平均耗时为 `t_i` 微秒，每次有效拷贝量为 `P_i` 字节：

- `avg_us_mean = mean(t_i)`。
- `avg_us_max = max(t_i)`：最慢 die 的平均耗时。
- `avg_bandwidth = sum(P_i / (t_i * 1000))`，单位 GB/s；不将读写流量翻倍。
- `P_i = batch_size * copy_count * bytes_per_token`；BF16 为 1152，C8 为 656；零拷贝带宽为 0。

硬件验收需覆盖四种组合。与原三个分支做性能回归时，用同一套独立源池输入、copy-cap、设备绑定、预热和计时条件交替测量至少五轮；历史 A5 表格采用共享源池，需在统一输入下重跑旧算子。存在可复现的性能下降时继续修复，通过后再删除旧分支。CPU 工具检查命令：`python3 -m unittest discover -s tests -p 'test_scatter_tools.py'`；它不替代 CANN 编译和 NPU 实测。
