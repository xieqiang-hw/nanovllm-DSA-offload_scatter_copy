# Ascend A3 KV-cache Scatter Copy

本分支仅包含 Ascend A3 上的 `NanovllmKvcacheScatterCopy` 算子、Torch 扩展以及单卡/多卡测试。

算子按照每个请求的 copy metadata，将 swapped-memory DRAM 中的 CKV/KPE token 搬运到 HBM cache。Block size 固定为 128，KPE dim 为 64，CKV dim 为 512，当前测试使用 BF16。

## 接口

```python
ops_overlap.kvcache_scatter_copy(
    hbm_kpe,              # bf16 [HBM_BLOCKS,128,64]，原地更新
    hbm_ckv,              # bf16 [HBM_BLOCKS,128,512]，原地更新
    dram_kpe,             # swapped-memory [DRAM_BLOCKS,128,64]
    dram_ckv,             # swapped-memory [DRAM_BLOCKS,128,512]
    hbm_block_table,      # int32 [B,HBM_BLOCKS_PER_ROW]
    dram_block_table,     # int32 [B,DRAM_BLOCKS_PER_ROW]
    source_token_ids,     # int32 [B,COPY_CAP]
    destination_slots,   # int32 [B,COPY_CAP]
    copy_counts,          # int32 [B]
) -> (hbm_kpe_alias, hbm_ckv_alias)
```

每行只读取 metadata 的前 `copy_counts[b]` 项；`copy_counts[b] == 0` 时该请求为 no-op。

## 编译

```bash
unset ASCEND_CUSTOM_OPP_PATH
unset OPS_OVERLAP_INSTALL_OPP_PATH
unset OPS_OVERLAP_OPC_SOC_VERSION
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
export CANN_INSTALL_PATH=$ASCEND_HOME_PATH
export SOC_VERSION=ascend910_9391
export OPS_OVERLAP_BUILD_JOBS=64
export OPS_OVERLAP_PYTHON=python3
export PYTHONUNBUFFERED=1
bash build.sh
```

`build.sh` 默认构建 `ascend910_9391`，同时支持通过 `SOC_VERSION` 选择 `ascend910b1`、`ascend910b3` 或其他 CANN 支持的 SoC。编译产物安装到仓库内的 `_custom_opp`，Torch 扩展原地生成在 `torch_extension/ops_overlap`。

编译完成后设置运行环境：

```bash
export ASCEND_CUSTOM_OPP_PATH=$PWD/_custom_opp/vendors/ops-overlap
export OPS_OVERLAP_INSTALL_OPP_PATH=$PWD/_custom_opp
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
```

## 单卡测试

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

python3 tests/test_scatter_copy.py \
  --device npu:0 --batch-size 24 \
  --source-len 65536 --hbm-slots 8192 \
  --copy-cap 2048 --copy-min 0 --copy-max 300 \
  --warmup 10 --iters 100 --seed 7
```

测试使用 `torch_npu.empty_with_swapped_memory` 创建真实 DRAM source，验证 DRAM→HBM 数据精确一致、输出 alias 和未命中 guard token，并使用 NPU Event 测量时延。测试全零 copy count 时设置 `--copy-min 0 --copy-max 0`。

## 多卡测试

```bash
bash tests/run_scatter_copy_multiprocess.sh
```

该批量脚本当前按 A3 测试机配置使用 `/usr/local/Ascend/cann-9.0.1` 和 `SOC_VERSION=ascend910_9391`；在其他安装路径运行时，应先相应调整脚本中的 CANN 路径。

多卡调度逻辑：

1. 每张 NPU 通过 `subprocess.Popen` 启动独立单卡 worker。
2. 所有 worker 完成数据构造和正确性检查后，通过 ready/start 文件同时开始 warmup。
3. 所有 worker 完成 warmup 后再次同步，同时开始 NPU Event 计时。
4. 父进程检查每个 worker 的返回码，读取单卡 JSON 并聚合结果。

批量脚本默认：

- 卡数：`1/2/4/8/16`
- batch size：`8/24`
- copy count：`0/100/200/300/500/2048`
- `source_len=65536`
- `hbm_slots=8192`
- `copy_cap=2048`
- `warmup=10`
- `iters=1000`
- `seed=7`，所有卡使用相同 workload

结果写入 `results/multiprocess`，每个 case 生成 JSON 和日志。全部 case 完成后自动运行分析脚本。

## 结果分析

```bash
python3 tests/analyze_scatter_copy_results.py \
  --results-dir results/multiprocess
```

分析依赖 `pandas`，输出：

- `scatter_copy_multiprocess_summary.json`
- `scatter_copy_timing_summary.csv`
- 控制台完整结果表和 pivot 表

`payload_gbps_sum` 是各卡独立带宽之和；`payload_gbps_sum_by_avg_us_max` 使用所有卡每轮 payload 总量除以最慢卡的平均时延，表示同步场景下的保守聚合吞吐。
