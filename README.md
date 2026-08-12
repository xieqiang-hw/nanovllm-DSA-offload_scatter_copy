# Ascend 950 DSA Offload Operators

本仓库包含四个 Ascend 950 原生算子：`kvcache_scatter_copy`、`sparse_and_tail_attention`、`sparse_and_tail_attention_and_scatter_copy` 和 `sparse_and_tail_attention_and_scatter_copy_mte_pipeline`。Attention 的 AIC/softmax/MM2 主流水复用并改造自 [vLLM-Ascend v0.23.0rc1 SparseFlashAttention Arch35](https://github.com/vllm-project/vllm-ascend/tree/v0.23.0rc1/csrc/attention/sparse_flash_attention)。基线融合算子在 AIV gather 阶段直接从 swapped-memory DRAM 读取 miss token，同时写入连续 Attention workspace 和 HBM KV cache。

MTE pipeline 是独立算子，不修改上述基线。它把 Attention 的虚拟 KV 顺序调整为 `local sparse hit -> local tail -> miss`：先计算不依赖 swapped-memory 的 hit tile，并在这些 ready tile 的计算窗口中分步预取 miss row；包含 miss 的 tile 尽量延后计算，从而拉长 swapped-memory 搬运与 Attention 计算的掩盖窗口。预取的数据同时写入 future Attention workspace 和 persistent HBM KV cache，避免 miss tile 再从 persistent cache 读取一次。

当前穿刺范围是 GLM-5.1 MLA decode：`q_seq_len=1`、Paged KV block size 128、CKV/KPE 维度 512/64。Attention 支持 bf16/fp16；独立 `kvcache_scatter_copy` 额外支持 int8。当前语义与性能验收配置固定为 TP16 对应的 `q_head=8`。

## 遗留问题

`q_head=128` 尚不支持。当前单 kernel split-G 移植在通过 `C=0` dense 阶段后，会卡在第一次 sparse Attention；官方 vLLM-Ascend A5 SFA 的单次 split-G 调度仍需继续完整移植。测试脚本会直接拒绝该配置，避免设备卡死，后续必须修复并补充 `q_head=128` 语义与时延验收。

## 接口

```python
torch.ops.ops_dsa_offload_a5.kvcache_scatter_copy(
    hbm_k_rope,          # bf16/fp16/int8[HBM_BLOCKS,128,64], in/out
    hbm_kv_cache,        # bf16/fp16/int8[HBM_BLOCKS,128,512], in/out
    dram_k_rope,         # bf16/fp16/int8[DRAM_BLOCKS,128,64], swapped memory
    dram_kv_cache,       # bf16/fp16/int8[DRAM_BLOCKS,128,512], swapped memory
    hbm_block_table,     # int32[B,HBM_MAX_BLOCKS]
    dram_block_table,    # int32[B,DRAM_MAX_BLOCKS]
    source_token_ids,    # int32[B,COPY_CAP]
    destination_slots,   # int32[B,COPY_CAP]
    copy_counts,         # int32[B]
) -> (hbm_k_rope, hbm_kv_cache)
```

```python
torch.ops.ops_dsa_offload_a5.sparse_and_tail_attention(
    query,                   # bf16/fp16[B,N,512]
    key,                     # bf16/fp16[HBM_BLOCKS,128,1,512]
    value,                   # 与 key 相同；MLA value alias
    sparse_slots,            # int32[B,1,2048]
    cache_tokens,            # int32[B], C；C=0 表示 dense 全量 KV
    block_table,             # int32[B,HBM_MAX_BLOCKS]
    actual_seq_lengths_query,# int32[B], TND 累计长度
    actual_seq_lengths_kv,   # int32[B], C+tail
    query_rope,              # bf16/fp16[B,N,64]
    key_rope,                # bf16/fp16[HBM_BLOCKS,128,1,64]
    scale_value,             # float
) -> attention_out           # bf16/fp16[B,N,512]
```

```python
torch.ops.ops_dsa_offload_a5.sparse_and_tail_attention_and_scatter_copy(
    query,                   # bf16/fp16[B,N,512]
    hbm_kv_cache,            # bf16/fp16[HBM_BLOCKS,128,1,512], in/out
    sparse_slots,            # int32[B,1,2048]；miss destination 在前
    cache_tokens,            # int32[B], C
    hbm_block_table,         # int32[B,HBM_MAX_BLOCKS]
    actual_seq_lengths_query,# int32[B], TND 累计长度
    actual_seq_lengths_kv,   # int32[B], C+tail
    query_rope,              # bf16/fp16[B,N,64]
    hbm_k_rope,              # bf16/fp16[HBM_BLOCKS,128,1,64], in/out
    dram_k_rope,             # bf16/fp16[DRAM_BLOCKS,128,64], swapped memory
    dram_kv_cache,           # bf16/fp16[DRAM_BLOCKS,128,512], swapped memory
    dram_block_table,        # int32[B,DRAM_MAX_BLOCKS]
    source_token_ids,        # int32[B,2048]；有效 miss source 在前
    copy_counts,             # int32[B]
    scale_value,             # float
) -> (attention_out, hbm_k_rope, hbm_kv_cache)
```

MTE pipeline 与基线融合算子保持相同的 Tensor 参数和返回值，只在最后增加一个可选调度参数：

```python
torch.ops.ops_dsa_offload_a5.sparse_and_tail_attention_and_scatter_copy_mte_pipeline(
    query,                   # bf16/fp16[B,N,512]
    hbm_kv_cache,            # bf16/fp16[HBM_BLOCKS,128,1,512], in/out
    sparse_slots,            # int32[B,1,2048]；miss destination 在前
    cache_tokens,            # int32[B], C
    hbm_block_table,         # int32[B,HBM_MAX_BLOCKS]
    actual_seq_lengths_query,# int32[B], TND 累计长度
    actual_seq_lengths_kv,   # int32[B], C+tail
    query_rope,              # bf16/fp16[B,N,64]
    hbm_k_rope,              # bf16/fp16[HBM_BLOCKS,128,1,64], in/out
    dram_k_rope,             # bf16/fp16[DRAM_BLOCKS,128,64], swapped memory
    dram_kv_cache,           # bf16/fp16[DRAM_BLOCKS,128,512], swapped memory
    dram_block_table,        # int32[B,DRAM_MAX_BLOCKS]
    source_token_ids,        # int32[B,2048]；有效 miss source 在前
    copy_counts,             # int32[B]
    scale_value,             # float
    prefetch_rows_per_step=5,# int，范围[0,16]
) -> (attention_out, hbm_k_rope, hbm_kv_cache)
```

`prefetch_rows_per_step` 表示每个 ready tile 最多推进多少个 swapped-memory row，默认值为 5；设为 0 可保留 hit-first 顺序但关闭分步预取。每请求不超过 400 个 miss 时，算子使用五个 `[128,576]` future-workspace tile；超过 400 个 miss 时回到原有 persistent-HBM/三 workspace 路径。

## 编译

Ascend 950 必须使用 CANN 9.1 环境；不要设置 CANN 8.5.1、`SOC_VERSION=ascend910_9391` 或 `OPS_OVERLAP_*`。`build.sh` 固定无交互编译本仓库的全部四个算子。

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset OPS_DSA_OFFLOAD_A5_INSTALL_OPP_PATH
unset OPS_OVERLAP_INSTALL_OPP_PATH
unset OPS_OVERLAP_OPC_SOC_VERSION
unset OPS_OVERLAP_BUILD_JOBS
unset OPS_OVERLAP_PYTHON
unset SOC_VERSION
unset CANN_INSTALL_PATH
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export A5_SOC_VERSION=ascend950
export OPS_DSA_A5_BUILD_JOBS=64
export OPS_DSA_A5_PYTHON=python3
export PYTHONUNBUFFERED=1
bash build.sh
```

## 单测

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset OPS_DSA_OFFLOAD_A5_INSTALL_OPP_PATH
unset OPS_OVERLAP_INSTALL_OPP_PATH
unset OPS_OVERLAP_OPC_SOC_VERSION
unset OPS_OVERLAP_BUILD_JOBS
unset OPS_OVERLAP_PYTHON
unset SOC_VERSION
unset CANN_INSTALL_PATH
unset IGNORE_INFER_ERROR
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export A5_SOC_VERSION=ascend950
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
python3 tests/test_dram_to_hbm.py --device npu:0 --dtype bf16 --batch-size 24 --source-len 20000 --hbm-slots 6144 --copy-min 0 --copy-max 300 --warmup 0 --iters 1 --seed 7
python3 tests/test_dram_to_hbm.py --device npu:0 --dtype int8 --batch-size 24 --source-len 20000 --hbm-slots 6144 --copy-min 0 --copy-max 300 --warmup 0 --iters 1 --seed 7
python3 tests/test_sparse_attention_scatter.py --device npu:0 --mode check --batch-size 24 --heads 8 --source-len 65536 --cache-tokens 8192 --tail-tokens 64 --miss-min 0 --miss-max 300 --seed 7
python3 tests/test_sfa_mte_pipeline.py --device npu:0 --mode check --batch-size 32 --heads 8 --source-len 65536 --cache-tokens 8192 --tail-tokens 64 --miss-min 0 --miss-max 300 --prefetch-rows 5 --seed 42
```

Attention 融合单测分别 poison 串行和融合路线的 HBM destination，验证 swapped-memory DRAM→HBM 搬运、未参与搬运的 guard、串行/融合输出以及独立 CPU Attention golden；出现 `FUSED_SCATTER_ATTENTION_UT_OK` 才表示语义穿刺通过。

## Scatter-Copy 算子：正确性与时延

`test_dram_to_hbm.py` 使用 `empty_with_swapped_memory` 构造真实 DRAM source，验证 CKV/KPE 精确搬运、输出 alias、未参与搬运的 HBM guard 和全零 `copy_counts` 不修改 cache，并使用 NPU Event 测量单卡时延。支持 bf16 和 int8。

`test_scatter_copy_multiprocess.py` 为每张 NPU 启动一个独立子进程。各子进程完成数据构造、正确性检查和零 `copy_counts` 检查后，通过 ready/start 文件统一开始，随后独立执行 warmup 和 NPU Event 计时。批量脚本使用 `--same-seed`，保证每张卡具有相同 workload。

```bash

# 多进程完整 sweep：1/2/4/8 卡，batch size 8/32，copy count 0/100/200/300/500/2048
bash tests/run_scatter_copy_multiprocess.sh

# 单 case msprof：默认 8 卡、bf16、batch size 32、copy count 200、iters 1000
bash tests/msprof.sh
```

批量脚本默认使用 bf16、`source_len=65536`、`hbm_slots=8192`、`copy_cap=2048`、`warmup=10`、`iters=1000` 和 `seed=7`，结果写入 `results/a5_multiprocess`。可通过 `RESULTS_DIR`、`CARD_COUNTS`、`BATCH_SIZES`、`COPY_COUNTS`、`DTYPES`、`SOURCE_LEN`、`HBM_SLOTS`、`COPY_CAP`、`WARMUP`、`ITERS` 和 `SEED` 覆盖默认配置。每个 case 生成 JSON 和日志，全部 case 完成后生成 `scatter_copy_multiprocess_summary.json`。

`msprof.sh` 默认将 JSON、日志和 msprof 数据写入 `results/a5_msprof_copy200_1000iters`；可通过 `RESULTS_DIR`、`DEVICES`、`DTYPE`、`BATCH_SIZE`、`COPY_COUNT`、`WARMUP`、`ITERS` 等环境变量覆盖配置。

结果中的 `payload_gbps_sum` 是各卡独立带宽之和；`payload_gbps_sum_by_avg_us_max` 使用所有卡每轮 payload 总量除以各卡 `avg_us` 的最大值，表示以最慢卡平均时延计算的聚合吞吐。

## q_head=8 性能测试

性能用例对每个请求从完整 source 序列无放回随机采样 miss token，并随机化 destination slot 及两侧 block table；脚本会拒绝连续源地址或连续目的地址。A5 统一使用至少 100 次 warmup。

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset OPS_DSA_OFFLOAD_A5_INSTALL_OPP_PATH
unset OPS_OVERLAP_INSTALL_OPP_PATH
unset OPS_OVERLAP_OPC_SOC_VERSION
unset OPS_OVERLAP_BUILD_JOBS
unset OPS_OVERLAP_PYTHON
unset SOC_VERSION
unset CANN_INSTALL_PATH
unset IGNORE_INFER_ERROR
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export A5_SOC_VERSION=ascend950
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
python3 tests/test_dram_to_hbm.py --device npu:0 --dtype bf16 --batch-size 24 --source-len 20000 --hbm-slots 6144 --copy-min 0 --copy-max 0 --copy-cap 2048 --warmup 100 --iters 100 --seed 7
python3 tests/test_dram_to_hbm.py --device npu:0 --dtype bf16 --batch-size 24 --source-len 20000 --hbm-slots 6144 --copy-min 0 --copy-max 300 --copy-cap 2048 --warmup 100 --iters 100 --seed 7
python3 tests/test_dram_to_hbm.py --device npu:0 --dtype bf16 --batch-size 24 --source-len 20000 --hbm-slots 6144 --copy-min 300 --copy-max 300 --copy-cap 2048 --warmup 100 --iters 100 --seed 7
python3 tests/test_dram_to_hbm.py --device npu:0 --dtype int8 --batch-size 24 --source-len 20000 --hbm-slots 6144 --copy-min 300 --copy-max 300 --copy-cap 2048 --warmup 100 --iters 100 --seed 7

for bs in 1 4 8 12 16 24; do python3 tests/test_sparse_attention_scatter.py --device npu:0 --mode bench --batch-size "$bs" --heads 8 --source-len 65536 --cache-tokens 8192 --tail-tokens 64 --miss-min 0 --miss-max 300 --warmup 100 --iters 100 --seed 7; done

for miss in 0 100 200 300 500 2000 2048; do python3 tests/test_sparse_attention_scatter.py --device npu:0 --mode bench --batch-size 24 --heads 8 --source-len 65536 --cache-tokens 8192 --tail-tokens 64 --miss-min "$miss" --miss-max "$miss" --warmup 100 --iters 100 --seed 7; done
```

每个 Attention 配置由独立 Python 进程执行，避免 ACLNN tiling cache 干扰。前三条分别测 SCATTER 的零搬运、随机 0～300 和固定 300；第一组循环观察 batch 扩展，第二组循环观察固定 miss count 下串行 `SCATTER+SFA` 与融合算子的时延。

## MTE pipeline 性能测试

MTE pipeline 使用相同的随机 source/destination block table 和 swapped-memory 数据布局。下面只运行 Event 性能测量；`--miss-min/--miss-max` 控制每个请求独立采样的 miss 数量，二者相等时表示固定 miss 负载。

```bash
python3 tests/test_sfa_mte_pipeline.py \
  --device npu:0 --mode bench \
  --batch-size 32 --heads 8 \
  --source-len 65536 --cache-tokens 8192 --tail-tokens 64 \
  --miss-min 300 --miss-max 300 \
  --prefetch-rows 5 --warmup 20 --iters 100 --seed 7
```

使用 msprof 扫描核心矩阵：

```bash
OUTPUT_ROOT=/tmp/a5_sfa_mte_msprof \
BATCH_SIZES="8 32" \
MISS_RANGES="0:0 100:100 200:200 300:300 500:500 2048:2048" \
PREFETCH_ROWS="5" \
PROFILE_REPLAYS=1 \
bash tests/run_sfa_mte_msprof_matrix.sh
```

每个 case 的 profile 位于：

```text
/tmp/a5_sfa_mte_msprof/
  bs_<BS>_qheads_8_miss_<MIN>_<MAX>_prefetch_<ROWS>/
    PROF_*/mindstudio_profiler_output/
```

profile 中依次重放独立 scatter、独立 SFA、基线融合算子和 MTE pipeline。`prefetch_rows=0` 可作为仅重排 hit/miss、不开启分步预取的对照组。

## 命中率95%，每请求加载100个token

时延单位：μs。

| BS | Copy 时延 | SFA 时延 | Copy+SFA 融合时延 |
|---:|---:|---:|---:|
| 1 | 45.822 | 104.509 | 137.087 |
| 2 | 47.291 | 109.871 | 134.572 |
| 5 | 53.878 | 136.937 | 146.925 |
| 8 | 60.256 | 110.688 | 144.442 |
| 11 | 67.307 | 112.080 | 153.514 |
| 16 | 76.461 | 112.090 | 154.251 |
