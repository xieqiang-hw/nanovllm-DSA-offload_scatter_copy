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

## 统一带宽测试

三个分支使用相同的入口、参数和六列结果。本分支测试 **A3 BF16**；每个有效 token 为 **1152 字节**，默认 `COPY_CAP=2048`。BF16 需要 HBM slots 大于实际 copy count，以保留 guard token。

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

BF16 继续验证真实 swapped-memory DRAM→HBM 数据精确一致、输出 alias 和未命中的 guard token。

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

默认输出目录为 `results/scatter_copy_a3`，各分支分开保存，防止切换分支后混入其他格式。每组生成 JSON 和日志，并生成统一命名的汇总文件：

- `scatter_copy_timing_summary.csv`：六列数据。
- `scatter_copy_timing_summary.md`：相同六列 Markdown 表。
- `scatter_copy_multiprocess_summary.json`：同口径、未舍入的结果行。
- `scatter_copy_run_manifest.json`：本轮 case 清单及完成状态。每次扫参只汇总本轮选择的 case；失败的重跑不会继续使用上次成功结果。

逐 case JSON 继续保存配置、每 die 耗时、有效字节数和正确性诊断，便于复核；不再输出其他定义的带宽。worker 的完整日志保存在对应 `.log`，失败时终端显示日志末尾。

重新分析已有结果（无需 NPU，全部只依赖 Python 标准库）：

```bash
python3 tests/analyze_scatter_copy_results.py \
  --results-dir results/scatter_copy_a3
python3 tests/format_scatter_copy_csv.py \
  --input results/scatter_copy_a3/scatter_copy_timing_summary.csv
```

旧 JSON 可使用 `--results-dir` 指向原目录；分析器从 `per_device` 的字节数和时间重新计算。没有本轮清单的旧目录会读取所有 `cards*.json`；若同一组 cards/BS/copy 出现多份结果，会报错并要求分开分析，避免静默合并不同配置。旧 CSV 需要先从原始 JSON 重新生成。A5 的旧 `run/test/analyze/format_kvcache_scatter_copy*` 入口仍可使用。

CPU 上的工具回归检查：

```bash
python3 tests/test_scatter_tools.py -v
```

该检查覆盖参数传递、非连续可见设备、逐 die 带宽求和、零 copy、有效字节数校验、六列表格、旧 JSON 读取和旧结果隔离；它不替代目标 A3/A5 上的算子正确性与带宽实测。

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
- 若 `move_pages` 无法解析，保留原始错误并尝试只读后备查询：读取当前进程 `/proc/self/pagemap` 的 sampled Host VA 页表项，检查 PFN 非零且 present，再确认其物理页属于 `/proc/iomem` 的 System RAM、对应 sysfs memory block 为 online 且唯一归属一个 NUMA node。不会把设备物理地址、缺失/遮蔽的 PFN、离线块或多节点块猜成 Host node；原查询部分成功时，两个方法的已知节点也必须一致。后备查询独立验证全部样本，不拼凑不同样本的计数。仅读取元数据，不读取 `/proc/self/mem`、不触碰 buffer 内容、不改权限、不迁移页、不改变分配器。
- **后备方法也可能不可用。** [Linux pagemap 文档](https://www.kernel.org/doc/html/v5.10/admin-guide/mm/pagemap.html) 说明，缺少所需 `CAP_SYS_ADMIN` 权限时 PFN 会被隐藏；而 [Linux 5.10 pagewalk](https://github.com/torvalds/linux/blob/v5.10/mm/pagewalk.c) 对 `PFNMAP` VMA 可能直接按 hole 处理，即使 root 也读不到实际页。脚本保存相关 `/proc/self/maps`、`numa_maps` 和 `smaps` 的 `VmFlags`（`pf` 表示 PFNMAP）用于区分原因。`pagemap` 未返回 present 不证明驱动 buffer 没有物理内存，`resident_pages=0` 也只表示查询未解析到节点。
- 如果未捕获可用映射或两种页查询仍不完整，结果保持 `unverified`。例如 `100000000000 bind:0 file=/dev/davinci_manager` 只显示整段驱动 VMA 的策略，没有 `N0=...` 等实际页计数，**不能证明 source 分配在 node0**。如果确认是驱动映射不可见，应继续调查匹配版本的驱动查询接口，不放宽严格校验、不通过 CPU 强行读取/迁移已注册页解决。地址采集依赖该版本 Torch 的 warning 格式，未来版本改变格式或未及时打屏时不会猜测后声称验证成功。
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

结果默认保存到 `results/numa/<时间>_<PID>/`（或 `--output-dir`）：`summary.json`、六列 `trials.csv`、包含策略/轮次/页位置诊断的 `trials.json`、每轮完整结果和各 worker 日志、测试前后硬件信息。统一的 `analyze_scatter_copy_results.py` 用于普通多卡扫参结果，不把不同 NUMA 策略混成一组。

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

worker 日志中的 `A3_SCATTER_NUMA_ADDRESS_CAPTURE` 显示采集的映射数；`A3_SCATTER_NUMA_BUFFER` 分别打印 CKV/KPE 的 Host/SVM 地址、`vm_flags`、所用查询 `method`、两种查询的错误、`pagemap_observations`、`node_pages`、`diagnosis` 和 `verified`。外层实验失败时会直接重放当前 worker 的这些摘要和异常末尾，不必再逐层找日志；完整证据仍保存于 `A3_SCATTER_NUMA_PLACEMENT`。

只有最终 `placement_status=sample_verified` 才进入正式对照。保持 `--require-numa-placement`：如果出现 `pf` 且 `PAGEMAP_NOT_PRESENT_OR_HIDDEN`，或 `PFN_ZERO_OR_REDACTED`，请贴出新的摘要，不要反复执行相同命令、去掉校验或据此判断 NUMA 好坏。这里的 10 次迭代只用于诊断，不用于判断性能或 NUMA 收益。本次只改 Python 探针，无需重新编译 SCATTER。
