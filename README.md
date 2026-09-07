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

## 单卡

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python3 tests/test_kvcache_scatter_copy.py --device npu:0 --batch-size 24 --source-len 65536 --hbm-slots 8192 --copy-min 0 --copy-max 300 --warmup 10 --iters 1000 --seed 7
```

每次先校验真实 DRAM→HBM、16384-token 大拷贝、caller-owned、零 copy 和所有未写区域，再计时；HBM 校验前重新写 poison，656 字节全部精确比较。缺少 swapped-memory allocator 直接报错，不用 HBM 冒充 DRAM；不设置性能达标断言。`--dtype` 只接受 `c8`，可省略。

## 多卡

延续 BF16 分支的脚本名和环境变量。每张卡使用独立进程、独立 DRAM/HBM tensor，初始化后和预热后各同步一次；默认扫描卡数 `8 4 1`、BS `8 32`、每请求 copy `100 200 300 500 2048`，warmup=10、iters=1000。请求之间共享只读 DRAM 物理池、HBM 目标互不重叠，保持 BF16 测试的数据构造方式。

```bash
A5_TEST_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tests/run_kvcache_scatter_copy_multiprocess.sh
A5_TEST_VISIBLE_DEVICES=0,1,2,3 CARD_COUNTS="1 2 4" BATCH_SIZES="24" COPY_COUNTS="0 300 2048" ITERS=1000 bash tests/run_kvcache_scatter_copy_multiprocess.sh
```

也可指定逻辑卡直接跑一组；`--devices` 是 `ASCEND_RT_VISIBLE_DEVICES` 内重编号后的逻辑序号，BS 和 copy 数均为每卡负载。`--same-seed` 让各卡使用相同索引分布。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python3 tests/test_kvcache_scatter_copy_multiprocess.py --devices 0,1,2,3 --batch-size 24 --source-len 65536 --hbm-slots 8192 --copy-min 300 --copy-max 300 --warmup 10 --iters 1000 --same-seed --seed 7 --output results/kvcache_scatter_copy_c8_multiprocess/cards4_c8_bs24_copy300.json
```

## 结果

默认保存到 `results/kvcache_scatter_copy_c8_multiprocess`，扫描脚本自动生成逐卡 JSON、汇总 CSV 和 Markdown 表；分析工具只依赖 Python 标准库，支持任意卡数、BS 和 copy 范围。

```bash
python3 tests/analyze_kvcache_scatter_copy_results.py --results-dir results/kvcache_scatter_copy_c8_multiprocess
python3 tests/format_kvcache_scatter_copy_csv.py --input results/kvcache_scatter_copy_c8_multiprocess/kvcache_scatter_copy_timing_summary.csv
```

计时沿用 BF16 分支：NPU Event 包围连续 iters 次调用，初始化、校验和 warmup 不计入；固定 metadata 重复搬运，不会变成零 copy，也不逐轮刷新硬件缓存。`payload_gbps` 单位是 **GB/s**：单卡为 `实际 copied_tokens × 656 / (avg_us × 1000)`；汇总同时给出各卡带宽之和及 `总 payload / 最慢卡时延`，表格采用后者。多卡同步释放计时，但不保证每次 DMA 严格同时执行，因此总带宽是并发负载估计，不是总线硬件计数器读数。
