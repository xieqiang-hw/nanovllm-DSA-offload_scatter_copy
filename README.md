# GLM-5.1 LI Manage、Scatter+SFA 融合算子开发

## LI Manage 算子接口

```python
torch.ops.ops_overlap.li_manage(
    query,                   # bf16/fp16[B, N, 128]，N=32 或 64
    key,                     # bf16/fp16[NUM_BLOCKS, 128, 1, 128]
    weights,                 # bf16/fp16[B, N]
    req_pool_entries,        # int32[B]，batch row -> request-pool row
    cache_slots,             # int32[POOL_SIZE, SOURCE_CAPACITY]，in/out；SOURCE_CAPACITY <= 2^21
    cache_tokens,            # int32[B]，每项 <= 16383
    candidate_lens,          # int32[B]，每项 <= 2^21-1（2097151）
    block_table,             # int32[B, MAX_BLOCKS]，MAX_BLOCKS <= 16384
) -> (
    source_token_ids,        # int32[B, 1, 2048]，完整 top-2048，miss 在前、hit 在后
    destination_slots,       # int32[B, 1, 2048]，完整 top-2048 的 HBM slot
    miss_counts,             # int32[B]
    cache_slots_alias,       # cache_slots 的更新后 alias
)
```

`li_manage_out` 语义相同，但由调用方额外传入 `source_token_ids`、`destination_slots` 和 `miss_counts` 三个持久输出 buffer，供 NPUGraph capture/replay 使用。

算子目录为 `li_manage`。序列长度不超过 262144 时使用精确的 18-bit index + 14-bit slot payload；更长时借用 FP32 score 低 3 bit 携带 index 高 3 bit。容量为 2^21 时，每个 request-pool row 的 `cache_slots` 占 8 MiB，每个活跃请求的 score workspace 也占 8 MiB。

## Scatter+SFA 融合算子接口

```python
torch.ops.ops_overlap.sparse_and_tail_attention_and_scatter_copy(
    query,                       # bf16/fp16[B, N, 512], 1<=B<=24, 1<=N<=128
    sparse_slots,                # int32[B, 1, 2048], top-2048 的 HBM slot；miss 前缀同时是写入目标
    cache_tokens,                # int32[B], 每个请求的 HBM cache token 数 C
    hbm_block_table,             # int32[B, HBM_MAX_BLOCKS]
    actual_seq_lengths_query,    # int32[B], TND query 累计长度
    actual_seq_lengths_kv,       # int32[B], C+tail
    query_rope,                  # bf16/fp16[B, N, 64]
    hbm_k_rope,                  # bf16/fp16[HBM_BLOCKS, 128, 1, 64], in/out
    hbm_kv_cache,                # bf16/fp16[HBM_BLOCKS, 128, 1, 512], in/out
    dram_k_rope,                 # bf16/fp16[DRAM_BLOCKS, 128, 64]
    dram_kv_cache,               # bf16/fp16[DRAM_BLOCKS, 128, 512]
    dram_block_table,            # int32[B, DRAM_MAX_BLOCKS]
    source_token_ids,            # int32[B, 2048]，前 copy_counts[b] 项是 DRAM source token
    copy_counts,                 # int32[B], miss count，范围 0..2048
    scale_value,                 # float, Attention scale
) -> (
    attention_out,               # bf16/fp16[B, N, 512]
    hbm_k_rope_alias,             # hbm_k_rope 的更新后 alias
    hbm_kv_cache_alias,           # hbm_kv_cache 的更新后 alias
)
```



## 编译

host tiling 或 kernel 有任何改动都必须完整重新编译：

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset OPS_OVERLAP_INSTALL_OPP_PATH
unset OPS_OVERLAP_OPC_SOC_VERSION
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
SOC_VERSION=ascend910_9391 OPS_OVERLAP_BUILD_JOBS=64 OPS_OVERLAP_PYTHON=python3 PYTHONUNBUFFERED=1 bash build.sh
```



## LI Manage 正确性与时延测试

每轮 warmup 和正式计时前都会从独立初始副本恢复 `cache_slots`，恢复与同步不计入 LI Manage 时延。

先验证 32/64 heads、完整 2048 个 source index、零 miss、乱序 request pool 和图重放，再测时延：

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset OPS_OVERLAP_INSTALL_OPP_PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
export SOC_VERSION=ascend910_9391
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
python3 tests/test_li_manage_scatter.py --device npu:0 --heads 32,64 --seed 7 --warmup 2 --iters 10 --graph-replays 3
python3 tests/test_li_manage_perf.py --device npu:0 --heads 32 --batch-sizes 1,4,8,12,16,24 --seq-lens 20096,65536,12288 --cache-tokens 6144 --miss-ranges 0:300 --warmup 3 --iters 100 --seed 7
```

### 18-bit/21-bit 边界

脚本验证 18-bit 边界、score-low3 编解码、完整 Top2048、重复更新和 request-pool 状态，并在相同输入上比较原版 LightningIndexer 与 LI Manage；二者时延差作为索引管理开销代理值。

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset OPS_OVERLAP_INSTALL_OPP_PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
export SOC_VERSION=ascend910_9391
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
python3 tests/test_li_manage.py --device npu:0 --heads 32,64 --seq-lens 262144,262272 --batch-size 1 --cache-tokens 6144 --miss-count 300 --warmup 1 --iters 3 --seed 7
python3 tests/test_li_manage.py --device npu:0 --heads 32,64 --seq-lens 262272 --batch-size 24 --cache-tokens 6144 --miss-count 300 --warmup 10 --iters 200 --seed 7
python3 tests/test_li_manage.py --device npu:0 --heads 32,64 --seq-lens 2097151 --batch-size 4 --cache-tokens 6144 --miss-count 300 --warmup 1 --iters 3 --seed 7
```



## Scatter-Copy 算子：正确性与时延

`test_scatter_copy.py` 不提供 mode 参数，每次固定验证真实 swapped-memory DRAM→HBM 搬运、输出 alias 和未修改 guard，并用 NPU Event 测量稳定时延。需要测全零 copy count 时直接设置 `--copy-min 0 --copy-max 0`。

`test_scatter_copy_multiprocess.py` 为每张 NPU 启动一个独立子进程。各子进程完成数据构造和正确性检查后，通过 ready/start 文件统一开始，随后独立执行 warmup 和 NPU Event 计时。每张卡使用相同 seed 时具有相同 workload。

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset OPS_OVERLAP_INSTALL_OPP_PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=4
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=/usr/local/Ascend/cann-8.5.1
export SOC_VERSION=ascend910_9391
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH

python3 tests/test_scatter_copy.py --device npu:0 --batch-size 24 --source-len 65536 --hbm-slots 8192 --copy-min 100 --copy-max 200 --copy-cap 2048 --warmup 10 --iters 100 --seed 7

# 多进程完整 sweep：1/2/4/8/16 卡，batch size 8/24，copy count 0/100/200/300/500/2048
bash tests/run_scatter_copy_multiprocess.sh

# 单 case msprof：默认 16 卡、batch size 24、copy count 200、iters 1000
bash tests/msprof.sh
```

批量脚本默认使用 `source_len=65536`、`hbm_slots=8192`、`copy_cap=2048`、`warmup=10`、`iters=1000` 和 `seed=7`，结果写入 `results/multiprocess`。每个 case 生成 JSON 和日志，全部 case 完成后生成 `scatter_copy_multiprocess_summary.json`。

`msprof.sh` 默认将 JSON、日志和 msprof 数据写入 `results/msprof_copy200_1000iters`；可通过 `RESULTS_DIR`、`DEVICES`、`BATCH_SIZE`、`COPY_COUNT`、`WARMUP`、`ITERS` 等环境变量覆盖配置。

结果中的 `payload_gbps_sum` 是各卡独立带宽之和；`payload_gbps_sum_by_avg_us_max` 使用所有卡每轮 payload 总量除以各卡 `avg_us` 的最大值，表示以最慢卡平均时延计算的聚合吞吐。



## Scatter+SFA 融合算子：B=24 时延测试

本节只测试 Scatter+SFA 融合方案，不测试 LI Manage。每个 miss count 使用独立 Python 进程，避免 ACLNN tiling cache 干扰。`100/200/300/500/2000/2048` 用于观察真实 DRAM 搬运量；需要测全零 miss 时直接设置 `--miss-min 0 --miss-max 0`。

结果中的 `serial_ms` 是串行基线 `Scatter→SFA` 的整体时延，`fused_ms` 是 Scatter+SFA 融合算子的时延；`scatter_ms` 和 `sfa_ms` 在 serial、fused 测量全部结束后独立计时，因此不会影响前两项结果。独立测量存在 Event 边界和 kernel 间隔差异，`scatter_ms+sfa_ms` 不要求与 `serial_ms` 完全相等。

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export SOC_VERSION=ascend910_9391
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH

for miss in 100 200 300 500 2000 2048; do
  python3 tests/test_fused_attention_scatter.py --device npu:0 --batch-size 24 --heads 8 --source-len 65536 --cache-tokens 8192 --tail-tokens 256 --miss-min 0 --miss-max "$miss" --warmup 10 --iters 100 --seed 7
  python3 tests/test_fused_attention_scatter.py --device npu:0 --batch-size 24 --heads 8 --source-len 65536 --cache-tokens 8192 --tail-tokens 256 --miss-min "$miss" --miss-max "$miss" --warmup 10 --iters 100 --seed 7
done
```

### 时延结果

`commit_7f9a9a` 使用 `empty_with_swapped_memory` 分配真实 DRAM source，全部语义检查通过；配置为 GLM-5.1、TP16、B=24、q_head=8、source_len=65536、cache_tokens=8192、tail_tokens=256，时延单位为 μs。Scatter 折算 H2D 带宽按实际 `copied_tokens × (512+64) × 2 bytes / scatter_ms` 计算。

| miss_count | 平均 miss_count | Scatter 折算 H2D 带宽（GB/s） | Scatter 单算子时延 | SFA 单算子时延 | 串行基线（Scatter→SFA） | Scatter+SFA 融合算子 | 加速比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0～100（随机） | 49.125 | 38.553 | 35.230 | 105.793 | 133.604 | 107.160 | 1.2468× |
| 100（固定） | 100.000 | 69.241 | 39.930 | 106.448 | 163.794 | 109.130 | 1.5009× |
| 0～200（随机） | 93.000 | 70.845 | 36.294 | 106.138 | 144.042 | 110.315 | 1.3057× |
| 200（固定） | 200.000 | 77.001 | 71.812 | 105.995 | 179.949 | 126.448 | 1.4231× |
| 0～300（随机） | 115.208 | 72.614 | 43.866 | 106.242 | 151.945 | 113.521 | 1.3385× |
| 300（固定） | 300.000 | 82.648 | 100.358 | 107.946 | 210.897 | 149.472 | 1.4110× |
| 0～500（随机） | 250.875 | 84.693 | 81.898 | 107.115 | 193.596 | 138.915 | 1.3936× |
| 500（固定） | 500.000 | 87.829 | 157.396 | 108.082 | 267.402 | 211.264 | 1.2657× |
| 0～2000（随机） | 1028.125 | 92.352 | 307.796 | 108.197 | 419.001 | 338.464 | 1.2379× |
| 2000（固定） | 2000.000 | 94.351 | 586.064 | 107.048 | 693.188 | 600.784 | 1.1538× |
| 0～2048（随机） | 1027.125 | 92.917 | 305.626 | 108.050 | 414.226 | 333.380 | 1.2425× |
| 2048（固定） | 2048.000 | 94.099 | 601.740 | 107.967 | 709.326 | 611.484 | 1.1600× |
