# Ascend 950 C8 KV-cache Scatter Copy 带宽测试

`scatter_copy_a5_c8` 分支只编译 C8 scatter；BF16 测试保留在 `scatter_copy_a5` 分支。算子 host、tiling、kernel 和 C++ 调用适配复制自 `ops_lidu_scattercopy_sfa_a5@a1c31a4`，不依赖原仓库、vLLM 或 LIM，可在本仓独立编译和运行。

## 接口

```python
import torch
import vllm_dsa_a5

torch.ops.vllm_dsa_a5.kvcache_scatter_copy_c8(
    hbm_kv_bytes,       # int8[HBM_BLOCKS,128,1,656]，caller-owned，原地写入
    dram_kv_bytes,      # int8[DRAM_BLOCKS,128,1,656]，empty_with_swapped_memory 申请，只读
    hbm_block_table,    # int32[B,HBM_MAX_BLOCKS]，请求逻辑块 -> HBM 物理块
    dram_block_table,   # int32[B,DRAM_MAX_BLOCKS]，请求逻辑块 -> DRAM 物理块
    source_token_ids,   # int32[B,1,16384]，各请求的逻辑源 token ID
    destination_slots, # int32[B,1,16384]，各请求的逻辑目标 HBM slot
    copy_counts,       # int32[B]，只搬运各行前 n 项；0 为 no-op，最大 16384（含 first-fill）
) -> None
```

所有 tensor 位于同一 NPU、连续存储。每个 token 按 656 字节原样搬运（512B FP8 latent + 128B RoPE + 16B scale），不解量化；目标 slot 不得重复。源容量最多 262144 tokens；`--copy-cap` 固定为 16384，与实际 copy 数、HBM 预算无关。

## 编译

在仓库根目录运行；需要 Ascend 950 对应的 CANN 环境、匹配的 PyTorch/torch_npu 和 setuptools。只生成 `VllmA5KvcacheScatterCopyC8`，安装到本仓 `_custom_opp_c8`，导入扩展时自动绑定该目录。

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset NANOVLLM_CUST_OPAPI_LIB
unset NANOVLLM_A5_INSTALL_OPP_PATH
export ASCEND_HOME_PATH=/usr/local/Ascend/cann
export CANN_INSTALL_PATH=$ASCEND_HOME_PATH
source "$ASCEND_HOME_PATH/set_env.sh"
export A5_SOC_VERSION=ascend950
export NANOVLLM_A5_OPS_PYTHON=python3
export NANOVLLM_A5_OPS_BUILD_JOBS=64
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
bash build_c8.sh
```

## 统一带宽测试

三个分支使用相同的入口、参数和六列结果。本分支测试 **A5 C8**；每个有效 token 为 **656 字节**，默认 `COPY_CAP=16384`。C8 的 metadata 容量固定为 16384；实际 copy count 可小于该值。

在仓库根目录、已安装本分支编译产物的 CANN/PyTorch 环境中运行：

```bash
TEST_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CARD_COUNTS="1 2 4 8" \
BATCH_SIZES="8 16 24 32" \
COPY_COUNTS="0 100 200 300 500 2048" \
  bash tests/run_scatter_copy_multiprocess.sh
```

只跑一组时，三个分支的命令完全相同：

```bash
bash tests/run_scatter_copy_multiprocess.sh \
  --cards 8 --batch-size 8 --copy-count 300
```

参数也支持多个值，例如 `--cards 1 2 4 --batch-size 8 24 --copy-count 200 300`。命令行参数优先于环境变量。`--dry-run`（或 `DRY_RUN=1`）只检查参数并打印每组命令，不加载 CANN/Torch、不启动 NPU，也不写结果文件：

```bash
bash tests/run_scatter_copy_multiprocess.sh \
  --visible-devices 2,3,6,7 --cards 1 2 4 \
  --batch-size 8 16 --copy-count 0 200 300 --dry-run
```

参数约定：

- `cards` / `CARD_COUNTS` 统计当前可见列表中的**逻辑 NPU/die 数**。每组使用前 N 个可见设备；每个 die 启动独立进程。在 A3 的 8 张物理卡、16 die 机器上，全机测试要设置可见列表 `0,1,...,15`（实际命令中展开全部编号）和 `CARD_COUNTS="1 2 4 8 16"`。
- `TEST_VISIBLE_DEVICES` 指定可见设备，`--visible-devices` 可覆盖。兼容本分支旧的 `A3_TEST_VISIBLE_DEVICES` 或 `A5_TEST_VISIBLE_DEVICES`；优先级为统一变量、旧变量、已有 `ASCEND_RT_VISIBLE_DEVICES`、分支默认值。
- `batch_size` 是**每 die** 的请求数；`copy_count` 是**每个请求**的固定搬移 token 数。每 die 每轮有效 payload = `batch_size × copy_count × bytes_per_token`。
- 默认卡数为可见数量内的 `1 2 4 8 16`；A3 默认可见 16 个 die，A5 默认 8 个。显式指定时支持任意正数卡数，不局限于这些档位。
- 三个分支默认 `BATCH_SIZES="8 16 24 32"`、`COPY_COUNTS="0 100 200 300 500 2048"`、`SOURCE_LEN=65536`、`HBM_SLOTS=8192`、`WARMUP=10`、`ITERS=1000`、`SEED=7`。`COPY_CAP` 是 metadata 容量，不是实际 copy 数。
- 其余统一参数：`--source-len`、`--hbm-slots`、`--copy-cap`、`--warmup`、`--iters`、`--seed`、`--results-dir`；同名大写下划线环境变量也可使用，输出目录环境变量为 `RESULTS_DIR`。Python 路径可用 `PYTHON_BIN` 指定。
- 统一扫参固定使用本分支 dtype，防止去掉 dtype 列后混合不同格式。若设置旧 `DTYPES`，只能填本分支的单个 dtype。
- 空列表、重复值、负数、超出可见数量的卡数、超出 source/HBM/copy-cap 的 copy 数，在启动 worker 前报错。

## 单组与单 die 入口

也可直接调用 Python launcher，`--cards N` 与 `--devices` 二选一；后者是可见列表内重编号后的逻辑索引。所有参数在三个分支上使用相同名称；省略 `--output` 时，按 cards/BS/copy 自动命名并写入本分支默认结果目录：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD/torch_extension${PYTHONPATH:+:$PYTHONPATH}"
python3 tests/test_scatter_copy_multiprocess.py \
  --cards 4 --batch-size 24 --copy-count 300 \
  --source-len 65536 --hbm-slots 8192 \
  --warmup 10 --iters 1000 --same-seed --seed 7 \
  --output results/manual/cards4_bs24_copy300.json

python3 tests/test_scatter_copy.py \
  --device npu:0 --batch-size 24 --copy-count 300 \
  --source-len 65536 --hbm-slots 8192 --warmup 10 --iters 1000
```

`--copy-count` 同时设置 `--copy-min/--copy-max`。旧的随机范围参数仍可用于正确性测试；分析这种旧结果时 `copy_count` 显示为 `min..max`，不会把上限冒充固定 copy 数。

C8 继续验证真实 DRAM→HBM、16384-token 大拷贝、caller-owned、零 copy 和所有未写区域；每次校验前重新写 poison，656 字节全部精确比较。

## 六列结果与带宽口径

终端结果表、CSV 和 Markdown 均只输出以下六列，顺序固定，时间/带宽保留三位小数：

```text
cards  batch_size  copy_count  avg_us_mean  avg_us_max  avg_bandwidth
```

对每个 die，先用 NPU Event 包围 `iters` 次 Scatter，将计时时间除以 `iters`，得到 `t_i`（μs）。`D_i` 是该 die **每轮实际有效搬移字节数**。定义：

```text
avg_us_mean  = mean(t_i)
avg_us_max   = max(t_i)
avg_bandwidth = sum(D_i / (t_i * 1000))   # GB/s，先逐 die 计算再求和
```

`avg_bandwidth` 虽含 avg，含义是基于各 die 平均耗时计算的**带宽之和**，不再除以 cards。只计算一次有效 payload，不把源读取与目标写入重复计数，不使用整块分配容量。BF16 的 `D_i = copied_tokens_i × 1152`；C8 的 `D_i = copied_tokens_i × 656`（512B FP8 latent + 128B RoPE + 16B scale）。零 copy 的带宽为 0。

`avg_us_max` 是各 die 平均耗时的最大值，不是某次调用的最大耗时。固定 metadata 会重复搬运；初始化、正确性校验及 warmup 不在计时区间。该带宽是此并发测试的有效吞吐估计，结果不代表总线硬件计数器读数。

默认输出目录为 `results/scatter_copy_a5_c8`，各分支分开保存，防止切换分支后混入其他格式。每组生成 JSON 和日志，并生成统一命名的汇总文件：

- `scatter_copy_timing_summary.csv`：六列数据。
- `scatter_copy_timing_summary.md`：相同六列 Markdown 表。
- `scatter_copy_multiprocess_summary.json`：同口径、未舍入的结果行。
- `scatter_copy_run_manifest.json`：本轮 case 清单及完成状态。每次扫参只汇总本轮选择的 case；失败的重跑不会继续使用上次成功结果。

逐 case JSON 继续保存配置、每 die 耗时、有效字节数和正确性诊断，便于复核；不再输出其他定义的带宽。worker 的完整日志保存在对应 `.log`，失败时终端显示日志末尾。

重新分析已有结果（无需 NPU，全部只依赖 Python 标准库）：

```bash
python3 tests/analyze_scatter_copy_results.py \
  --results-dir results/scatter_copy_a5_c8
python3 tests/format_scatter_copy_csv.py \
  --input results/scatter_copy_a5_c8/scatter_copy_timing_summary.csv
```

旧 JSON 可使用 `--results-dir` 指向原目录；分析器从 `per_device` 的字节数和时间重新计算。没有本轮清单的旧目录会读取所有 `cards*.json`；若同一组 cards/BS/copy 出现多份结果，会报错并要求分开分析，避免静默合并不同配置。旧 CSV 需要先从原始 JSON 重新生成。A5 的旧 `run/test/analyze/format_kvcache_scatter_copy*` 入口仍可使用。

CPU 上的工具回归检查：

```bash
python3 tests/test_scatter_tools.py -v
```

该检查覆盖参数传递、非连续可见设备、逐 die 带宽求和、零 copy、有效字节数校验、六列表格、旧 JSON 读取和旧结果隔离；它不替代目标 A3/A5 上的算子正确性与带宽实测。
