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

## NUMA 亲和性与负载均衡实验

新增 `tests/benchmark_scatter_copy_numa.py`，不修改 SCATTER 内核、接口或 `empty_with_swapped_memory` 分配器。只涉及 Python 测试；已有本仓编译产物时无需重新编译，运行时沿用成功编译时的 CANN 环境。多卡 launcher 自动将本仓 `torch_extension` 加入子进程的 `PYTHONPATH`。

### 为什么值得测

worker-53215 的采集信息显示：4 个 CPU socket、每 socket 80 核/160 线程、共 640 个逻辑 CPU；Host 有 **8 个 NUMA 节点**，各约 252 GiB，而不是 4 个节点。CPU NUMA 距离为本地 10、成对节点之间 15、其它节点之间 20。NPU 则有 8 卡 × 2 die = 16 个软件设备。

`npu-smi topo` 中的 SIO/HCCS_SW 是 **NPU→NPU** 拓扑，没有给出 **NPU→Host NUMA node** 的亲和性。因此不能据此认定 Host DRAM 完全等距，也不能简单硬编码 `NPU id / 2 → NUMA node`。即使 NPU 访问各节点近似等距，内存集中到少数节点仍可能使控制器/链路负载不均。

该机器的 torch_npu git 为 `12d689a08941d4a6e45eab16e3ad6fef96a9affd`。[该版本分配器源码](https://github.com/Ascend/pytorch/blob/12d689a08941d4a6e45eab16e3ad6fef96a9affd/torch_npu/csrc/core/npu/NPUSwappedMemoryAllocator.cpp) 通过 CANN 分配 Host 内存，再注册为 NPU 可访问地址；没有显式传入 NUMA 节点。**这并不证明底层 CANN/驱动一定遵循或一定忽略 Linux NUMA 策略，必须实测页位置。** 此外，shell 的 `policy: default` 只代表该 shell 的默认策略，不代表已运行的 16 个 worker 的内存分布；各节点总 free memory 接近也不是 KV buffer 均衡的证据。

### 对照方法

| 策略 | 每个 worker 的内存策略 | 目的 |
|---|---|---|
| `default` | 显式恢复 Linux 默认策略 | 观察固定 CPU 绑核下的自然分配；不是强制均衡 |
| `concentrated` | 所有 worker 绑定 `--concentrated-node`，默认 0 | 人为集中分配的对照 |
| `balanced` | worker 按 `--nodes` round-robin 单节点绑定 | 16 worker / 8 节点时，每节点分配 2 个 worker |
| `interleave` | 每个 worker 在所选节点间按页交错分配 | 每份 source 都跨节点分布的另一种均衡方法 |

CPU 与内存策略分开控制：默认 `--cpu-bind spread --cpus-per-worker 4`，从当前允许的 CPU 中给每个 worker 分配不重叠的固定 CPU 集合；**四组实验保持该集合不变**。它不声称是 NPU 最近的 CPU，也不把集中内存的 worker 全部绑到 node 0 的 CPU。运行时线程池统一为 1 线程。若使用 `--cpu-bind none`，所有组都不主动绑核，CPU 调度噪声会更大。

每个策略/轮次都启动新进程，在导入 Torch/CANN **之前**设置 CPU affinity 和 NUMA 内存策略（包含进程的其它后续 Host 分配，不是只绑定两个 tensor）。[Linux 内存策略按线程继承](https://www.kernel.org/doc/html/v5.15/admin-guide/mm/numa_memory_policy.html)，导入后再只设置主线程可能遗漏 runtime 分配线程。设置失败直接报错；不悄悄降级。

所有 worker 使用相同 seed、source、metadata 和 copy count；warmup 后通过共同的未来时间点开始计时，记录 Host 起跑偏差。较快 worker 完成自己的计时后继续做不计时的拷贝，直到所有 worker 都完成计时，避免慢卡末段因快卡退出而失去竞争。实验默认 `--timing graph`，只 capture 一个 SCATTER，使用 NPU Event 测量重复 replay；`--timing eager` 可与原测试路径对照，不做自动 fallback。

采样位置发生在 correctness 检查后、warmup/计时之前，不在热路径：

- 分别查询真实 swapped CKV/KPE 的页位置，并另采样实际被 copy 的 token 所在页；不以普通 CPU golden tensor 或全进程 RSS 冒充 source 的位置。
- `move_pages(..., nodes=NULL, flags=0)` **只查询，不迁移、不解引用 NPU 指针、不修改已注册 DMA 页**。
- NUMA probe 会在导入 Torch/CANN 前将该 worker 的日志级别设为 WARNING、启用日志打屏。在本次 source 分配期间临时捕获原生 stdout/stderr，从 `torch_npu::registerSvmMem` 的 warning 中读取 `svmPtr → alignedPtr` 映射，随后原样重放捕获的日志。只采信当前 PID、与活跃 tensor 地址完全一致且无冲突的记录；不会读取旧 PLOG、硬编码上次运行的地址、按 VMA 大小猜测地址，也不修改/替换分配器。采集在 warmup/计时之前完成，各策略使用相同日志配置；不启用 NUMA probe 的旧测试不改变日志配置。
- `alignedPtr` 是注册区域的真实 Host 起点，页采样和实际 copy token 的偏移均基于此地址。查询前要求 `/proc/self/maps` 覆盖整个 Host buffer；如果驱动使用相同 Host/NPU VA，也支持在完整 Host VMA 上直接查询 tensor 地址。**有地址映射不等于页位置已经验证。**
- 如果当前运行未捕获可用映射、驱动 Host 映射无法查询、系统不允许 `move_pages`，结果明确标记 `unverified`，保存相关 `/proc/self/maps`、`numa_maps` 行和错误码。此时设置成功也不能证明实际 source 被集中或均衡分配。采集依赖该版本 Torch 的 warning 格式，未来版本若改变格式或没有及时打屏，不会猜测后继续声称验证成功。
- `sample_verified` 只表示抽样成功且符合指定节点，不是全量页扫描。`policy_mismatch` 表示已查到页却落在指定节点外；`interleave_not_observed` 表示样本未覆盖全部指定节点。可加 `--require-numa-placement`，要求验证通过后才能计时；默认保留诊断和时延，即使页位置未验证。
- `SCATTER_NUMA_NODE_LOAD` 根据样本打印按字节加权的 source 容量、实际 copy 读负载分布估计值；未验证时为 `null`，不会编造节点负载。

### 先测多卡集中分配 vs 均衡分配

**请在目标 NPU 和 Host 没有其它推理/拷贝负载时运行。** 这次采集时 16 个 die 都有推理进程，不能直接用当时的忙碌状态做性能比较。脚本会保存测试前后的 `npu-smi`、`lscpu` 和 NUMA 信息，但不会终止任何已有推理任务。

在已能运行本仓 SCATTER 测试的环境中，确认 16 个 NPU 可见，然后运行：

```bash
python3 tests/benchmark_scatter_copy_numa.py \
  --devices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --nodes 0-7 \
  --batch-size 12 --source-len 65536 --hbm-slots 8192 \
  --copy-count 300 --copy-cap 2048 \
  --warmup 10 --iters 2000 --rounds 3 --seed 7
```

`--devices` 是当前 `ASCEND_RT_VISIBLE_DEVICES` 列表内的逻辑索引；`--nodes` 是 Host NUMA 节点号，二者不要混淆。`copy-count=300` 表示**每个请求 300 个唯一 token 搬移**，bs=12 时每 die 每轮共 3600 个 token；不乘以 MTP 的 query 数。

默认运行四种策略，轮次间轮换顺序；每组重新分配相同 workload。可以先附加 `--dry-run` 只检查映射，不初始化 NPU。想先缩小测试时可用 `--policies concentrated,balanced --rounds 1`，或减少 `--devices`。

### 再查是否存在亲和性

如果分配位置能被验证，可用单 NPU 逐节点扫描，固定 CPU affinity，先排除多卡竞争：

```bash
python3 tests/benchmark_scatter_copy_numa.py \
  --experiment affinity --devices 0 --nodes 0-7 \
  --batch-size 12 --source-len 65536 --hbm-slots 8192 \
  --copy-count 300 --warmup 10 --iters 2000 --rounds 3 --seed 7
```

把 `--devices` 扩为多个 die 可形成完整矩阵；该模式**逐个**测试 NPU×node 组合，不把多个设备同时运行而污染亲和性判断。发现可靠映射后，可用底层 `test_scatter_copy_multiprocess.py --numa-policy mapped --numa-node-map 0,0,1,1,...` 测试指定方案，节点列表必须逐 worker 填写，不能真的写省略号。

### 看哪些结果

结果默认保存到 `results/numa/<时间>_<PID>/`（或 `--output-dir`）：`summary.json`、`trials.csv`、每轮完整结果和各 worker 日志、测试前后硬件信息。旧的 `analyze_scatter_copy_results.py` 继续用于原多卡扫参结果，不把不同 NUMA 策略混成一组。

优先比较：实际页分布是否改变；各轮 `rank_max_us`（最慢 die）、`rank_mean_us`、按最慢 die 计的聚合带宽；各 die 时延是否均衡；`host_start_skew_us` 是否远小于整个计时窗口。Host interval overlap 只描述计时区间，不是硬件链路利用率；快 worker 计时结束后仍保持 copy 负载。Host window 带宽是包含提交/同步开销的辅助量，不是物理 DRAM 总线带宽。

若 concentrated 慢而 balanced/interleave 稳定变快，且 source 页位置确实改变，才支持继续验证均衡分配收益。若单卡节点扫描出现稳定差异，再研究亲和映射。若页位置未验证、策略未改变实际页分布或背景负载不一致，不能由这组结果得出 NUMA 无影响/有收益的结论。当前 workload 会重复访问同一组 source token；它是受控的固定访问模式，不等同于整网 78 层的缓存冷热状态，也不能直接预测 COPYSFA 的掩盖率。

CPU-only 检查（不需要 NPU/编译）：

```bash
python3 tests/test_numa_support.py -v
```

### Host/SVM 地址不同的快速复测

`worker-53232` 已确认 `empty_with_swapped_memory` 返回的 SVM 地址与 Host 地址不同：旧探针对 tensor 地址直接查询时全部返回 `EFAULT`。这不是 SCATTER 拷贝错误，也不是 NUMA 绑定失败的证据。更新 Python 脚本后无需重新编译，先在空闲设备上运行短测：

```bash
python3 tests/benchmark_scatter_copy_numa.py \
  --experiment affinity --devices 0 --nodes 0,4 \
  --batch-size 12 --source-len 65536 --hbm-slots 8192 \
  --copy-count 300 --warmup 1 --iters 10 --rounds 1 \
  --require-numa-placement
```

worker 日志中的 `A3_SCATTER_NUMA_ADDRESS_CAPTURE` 显示采集的映射数；`A3_SCATTER_NUMA_BUFFER` 分别打印 CKV/KPE 的 `tensor_ptr`、`host_ptr`、`node_pages`、`page_errors` 和 `verified`。只有最终 `placement_status=sample_verified` 才进入正式对照；如果转换成功但 Host 页仍返回 `EFAULT`，保留严格校验，将完整 `A3_SCATTER_NUMA_PLACEMENT` 贴出，再调查驱动映射的可查询性。这里的 10 次迭代只用于诊断，不用于判断性能或 NUMA 收益。
