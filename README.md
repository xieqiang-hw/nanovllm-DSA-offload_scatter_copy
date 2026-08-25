# Ascend 950 KV-cache Scatter Copy

本仓库仅包含 Ascend 950 上的 `A5KvcacheScatterCopy` 算子、Torch 扩展以及单卡/多卡测试。

该算子按照每个请求的 copy metadata，将 swapped-memory DRAM 中的 CKV/KPE token 搬运到 HBM cache。Block size 固定为 128，KPE dim 为 64，CKV dim 为 512，支持 BF16 和 FP16。

## 接口

```python
nanovllm_dsa_a5.kvcache_scatter_copy(
    hbm_kpe,              # bf16/fp16 [HBM_BLOCKS,128,64]，原地更新
    hbm_ckv,              # bf16/fp16 [HBM_BLOCKS,128,512]，原地更新
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
unset NANOVLLM_A5_INSTALL_OPP_PATH
unset NANOVLLM_CUST_OPAPI_LIB
unset A5_SOC_VERSION
unset SOC_VERSION
unset CANN_INSTALL_PATH
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export CANN_INSTALL_PATH=$ASCEND_HOME_PATH
source "$ASCEND_HOME_PATH/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_LAUNCH_BLOCKING=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/torch_extension:$PYTHONPATH
export SOC_VERSION=ascend950
export NANOVLLM_A5_OPS_PYTHON=python3
export NANOVLLM_A5_OPS_BUILD_JOBS=64
bash build_bf16.sh
```

编译完成后可显式设置本地 OPP：

```bash
export ASCEND_CUSTOM_OPP_PATH=$PWD/_custom_opp_bf16/vendors/customize
export NANOVLLM_A5_INSTALL_OPP_PATH=$PWD/_custom_opp_bf16
export NANOVLLM_CUST_OPAPI_LIB=$PWD/_custom_opp_bf16/vendors/customize/op_api/lib/libcust_opapi.so
```

## 单卡测试

```bash
python3 tests/test_kvcache_scatter_copy.py \
  --device npu:0 --dtype bf16 --batch-size 24 \
  --source-len 65536 --hbm-slots 4096 \
  --copy-cap 2048 --copy-min 0 --copy-max 300 \
  --warmup 3 --iters 20 --seed 7
```

测试包含 swapped-memory DRAM→HBM 数据精确校验、输出 alias 校验、未命中 guard row 校验、零 copy no-op 校验和 NPU Event 性能计时。

## 多卡测试

```bash
bash tests/run_kvcache_scatter_copy_multiprocess.sh
```

多卡脚本为每张 NPU 启动独立 worker，并在初始化后、warmup 后分别同步，确保所有卡同时进入计时阶段。默认遍历 1/2/4/8 卡及多组 batch/copy count，结果保存到 `results/kvcache_scatter_copy_multiprocess`。

可通过环境变量缩小测试范围：

```bash
CARD_COUNTS="2 4" BATCH_SIZES="24" COPY_COUNTS="0 300 2048" ITERS=100 \
  bash tests/run_kvcache_scatter_copy_multiprocess.sh
```

单独分析已有结果：

```bash
python3 tests/analyze_kvcache_scatter_copy_results.py \
  --results-dir results/kvcache_scatter_copy_multiprocess
```

分析依赖 `pandas`，输出汇总 JSON、CSV、普通表和 pivot 表。
