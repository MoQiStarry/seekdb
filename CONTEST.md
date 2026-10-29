# seekdb × AMD hipVS —— CAGRA 索引构建优化：提交说明

本文件是本赛题要求的三项提交产物之一的入口：

1. **实现代码**：`tools/hipvs_bridge/seekdb_cuvs_bridge.c` 的一行参数改动（见 §1）；
2. **可复现的测试方法**：`contest/` 下的脚本 + §3/§4 的计时口径，与赛题 §6 一致；
3. **CPU/GPU 构建性能对比报告**：§6（GPU 侧有实测数据；CPU `lib=vsag` 侧尚未测出，原因与复现入口均写明）。

证据等级标记：**[实测]** 有逐轮原始日志；**[推算]** 由实测按公式计算；**[未验证]** 需要正式数据或生产路径才能证明。

---

## 0. 基线与提交

| 项 | 值 |
|---|---|
| 冻结基线 commit | `634888d9b037153fa143d2c913ba4b25503e1d62`（tag `amd-cagra-contest-baseline-v5`） |
| 本文件所在分支 | `amd-cagra-submit-v5` |
| 分支拓扑 | 由冻结 commit 直接派生，`git merge-base --is-ancestor 634888d9b HEAD` 为真 |

> 仓库内另有两条对照分支：`contest-baseline-safe` 只含本文档与脚本、**不含 §1 的代码改动**
> （冻结基线保底提交）；`amd-cagra-submit-v5-dedup` 在本文档基础上追加了快照 dataset 去重
> （未完成配对基准与重启恢复验证，默认不用于提交）。本文所有结论均以 `amd-cagra-submit-v5` 为准。

## 1. 实现

### 1.1 改动：NN-Descent 迭代次数减半

位置：[`tools/hipvs_bridge/seekdb_cuvs_bridge.c:38`](tools/hipvs_bridge/seekdb_cuvs_bridge.c#L38)

```c
-  h->ip->nn_descent_niter = 100;
+  h->ip->nn_descent_niter = 50;
```

基线 CAGRA 建图参数为 `build_algo=NN_DESCENT`、`graph_degree=128`、
`intermediate_graph_degree=256`、`nn_descent_niter=100`。NN-Descent 迭代是建图
关键路径上唯一随迭代次数线性增长的部分，迭代减半直接缩短 GPU 建图时间，且不改动：

- 图度（`graph_degree` / `intermediate_graph_degree`）——决定图体积与检索代价；
- 检索参数（`itopk_size=max(512, topk*4)`、`search_width=4`）——决定检索召回与吞吐；
- 快照格式、DDL 流程、SQL 语义、CPU `lib=vsag` 路径。

### 1.2 为什么只改这一个参数

建图时间与召回质量共用同一组参数。对 250K 正式数据的召回下降幅度没有测量手段时
（开发数据只有 10K×128 且召回饱和在 1.0），改动面越小、越容易界定风险：

- 降 `graph_degree` 会同时改变图体积、检索吞吐与召回，收益大但风险也大；
- 换 `build_algo`（`IVF_PQ` / `AUTO_SELECT`）改变了算法本身，召回行为不可预测；
- 检索参数改动会动查询侧结果，而查询吞吐本身有 90% 硬门槛。

因此本提交只保留「建图关键路径线性项减半」这一处改动。其余方向在
`contest/` 的实验记录中保留为未采用候选。

## 2. 编译

评测机环境：ROCm 7.2.4、GCC 13.3、CMake 3.31、Rust 1.97，hipVS 位于 `/opt/hipvs`。

```bash
# 1) bridge 动态库
bash tools/hipvs_bridge/build.sh "$PWD/build/bridge/libseekdb_cuvs_bridge.so"

# 2) observer
./build.sh release \
  -DOB_BUILD_CUVS=ON \
  -DOB_BUILD_CUVS_TRACE=OFF \
  -DCUVS_BRIDGE_LIB="$PWD/build/bridge/libseekdb_cuvs_bridge.so" \
  --make -j64
```

产物：`build_release/src/observer/seekdb` 与 `libseekdb_cuvs_bridge.so`。

注意：`src/oblib/lib/CMakeLists.txt:93` 的条件是
`if (OB_BUILD_CUVS AND EXISTS "${CUVS_BRIDGE_LIB}")`，**不传 `-DCUVS_BRIDGE_LIB`
不会报错，只会静默不链接 GPU 后端**，随后建索引会走 CPU VSAG。本仓库的编译说明
（[`docs/gpu-vector-index-hipvs-cuvs/README.md`](docs/gpu-vector-index-hipvs-cuvs/README.md)）
与该文件保持一致的命令。

## 3. 计时口径（与赛题 §6 一致）

- 起点：原始向量已 `LOAD DATA` 导入、无向量索引、无可复用 CAGRA 图/索引文件/进程内缓存，
  计时从 `CREATE VECTOR INDEX ... lib=cuvs` 开始；
- 终点：索引创建完成、GPU 建图同步完成、**第一条固定查询返回合法结果**；
- 判定「确实在 GPU 上建图」：日志出现且仅出现一条
  `[cuVS] serialized GPU CAGRA snapshot(... built=true, build_us=...)`；
  `deserialized` 不参与建图判定。

`build_us` 是 bridge 内部 CAGRA build + 图序列化耗时；`ddl_wall_ms` 是客户端
DDL 往返墙钟（含 DDL 固定开销，不含首查询）。两者都记录，用于区分参数收益与固定开销。

## 4. 可复现的测试方法

### 4.1 数据

- 10K×128 开发集：赛题 Notebook 提供（`base.f32`、`query.f32`、`gt_100x10.i32`）；
- 或自建同构数据：`python3 contest/gen_dev_dataset.py --out <dir> --n 10000 --dim 128 --nq 4096`;
- 正式数据（SIFT250K 250K×128、Hidden250K 250K×256）由主办方在评测机提供，本地不可得。

### 4.2 单轮基准

```bash
# GPU 路径（lib=cuvs）
SEEKDB_OBS=/path/to/observer \
BASE_F32=/data/base.f32 QUERY_F32=/data/query.f32 GT_I32=/data/gt_100x10.i32 \
DIM=128 N=10000 NQ=100 PORT=2884 OUT_ROOT=/tmp/bench \
  bash contest/run_build_bench.sh V1-cuvs cuvs /path/to/bridge_dir
```

`contest/run_build_bench.sh` 每轮使用全新 base-dir 与独立端口，依次完成：
启动 observer → 建表 → 导入 → 计时 DDL 建索引 → 首条查询 → 召回评估
→ 采样 VRAM（`rocm-smi`，1s）与 `VmHWM` → 优雅停止 → 写出 `metrics.json`。

### 4.3 配对与重复

只有同一实例、同一轮次的「基线 vs 候选」比值可用。做法：两个版本交替运行 ≥3 轮
（`contest/run_cpu_gpu_compare.sh` 给出交替脚本骨架），比较 `build_us`、`ddl_wall_ms`
的逐轮比值，取几何均值。不同轮次之间的绝对毫秒会漂移（本机观测到同版本轮间
2.1× 波动，见 §5.3），不可跨轮比较绝对值。

### 4.4 查询口径

正式查询为 nq=4096 条单行 SQL，分别 Top-10 / Top-100，统计 QPS 与 Recall。
本提交未改检索代码，但图结构变化会影响检索路径，因此查询门槛需在正式数据上重新确认（§7）。

## 5. 实测结果

### 5.1 10K×128 建索引，GPU 路径，V0 与 V1 交替 3 轮 **[实测]**

环境：AMD Radeon（48GB，ROCm 7.x），同一实例同一时段；原始日志在
`contest/evidence/10k-gpu/`（每轮 `metrics.json`、`build.log`、`recall10.json`、
`bridge.sha256`、`vram.samples`）。

| 轮次 | V0 `build_us` | V1 `build_us` | 比值 | V0 DDL (ms) | V1 DDL (ms) | 比值 |
|---|---|---|---|---|---|---|
| r1 | 6,055,339 | 3,529,664 | 0.583 | 6,870 | 4,334 | 0.631 |
| r2 | 8,135,782 | 4,230,031 | 0.520 | 8,929 | 5,056 | 0.566 |
| r3 | 12,751,194 | 7,168,184 | 0.562 | 13,544 | 8,051 | 0.594 |
| 几何均值 | — | — | **0.554** | — | — | **0.597** |

即 10K 规模上 GPU 建图耗时约 **1.80×** 加速（DDL 墙钟 **1.68×**）。
召回：6 轮 Recall@10 均为 **1.000**；`approx` 与 `exact` 查询结果逐条一致；
`graph_bytes` 6 轮均为 10,241,713（图结构参数未变）。
显存峰值 1.63–1.68 GB，`VmHWM` 1.05–1.12 GB，与基线同量级。

这些数字只覆盖**开发规模**（10K×128）。正式 250K 的固定开销占比更小，
但召回下降无法在饱和数据上观测，见 §7。

### 5.2 50K 代理集上的参数方向（探针级，非生产路径）**[实测，代理]**

用链接同一 `libcuvs_c.so` 的独立探针在合成数据上做同轮配对：

| 变体 | build 配对比值 256d / 128d | 代理 Recall@10（256d / 128d） |
|---|---|---|
| 基线（niter=100） | 1.000 / 1.000 | 0.728 / 0.897 |
| niter=50（本提交） | 0.601 / 0.565 | 0.841 / 0.954 |
| niter=15 | 0.252 / 0.257 | 1.000 / 0.9997 |

同时用图结构度量（真 top-10 成为邻居的比例、长程边占比）检查图质量，得到与代理召回
**相反**的排序：niter 越大图越接近真 kNN（precision@128 从 0.049 升到 0.755），
但在合成「多簇 + 簇内近重复」数据上代理召回反而下降。结论：**代理召回不能用于判定
niter 方向**，正式召回只能在 250K 数据上测。这也是本提交不采用 niter=15/25 的原因。

### 5.3 环境噪声 **[实测]**

同一版本在不同轮次的绝对耗时相差可达 2.1×（外部 GPU 争用，`rocm-smi` 持续 100%）。
因此 §5.1 只报同轮交替的逐轮比值，不报跨轮绝对值。

## 6. CPU（`lib=vsag`）与 GPU（`lib=cuvs`）构建性能对比

### 6.1 方法

同一份数据、同一 observer 二进制、同一实例，分别执行：

```sql
-- CPU
CREATE VECTOR INDEX idx ON items(v) WITH (distance=l2, type=hnsw, lib=vsag);
-- GPU
CREATE VECTOR INDEX idx ON items(v) WITH (distance=l2, type=hnsw, lib=cuvs);
```

记录 DDL 墙钟、GPU 侧 `build_us`、Recall@10、显存/内存峰值。
脚本：`contest/run_cpu_gpu_compare.sh`（交替运行两者并汇总比值）。

### 6.2 当前数据

| 路径 | 10K×128 建索引耗时 | 来源 |
|---|---|---|
| GPU `lib=cuvs`，niter=100（冻结基线） | 6.06 / 8.14 / 12.75 s | §5.1 **[实测]** |
| GPU `lib=cuvs`，niter=50（本提交） | 3.53 / 4.23 / 7.17 s | §5.1 **[实测]** |
| CPU `lib=vsag` | **未测** | 无 Linux/AMD 环境，正式数据不在本地 **[未验证]** |

**CPU 侧没有数字，本报告不填推断值。** 所需环境与命令已在 §4 与
`contest/run_cpu_gpu_compare.sh` 给出；在任意一台安装 ROCm + hipVS 的 Linux 机器上
用 10K 数据约 15 分钟即可补齐。

### 6.3 如何补齐 CPU/GPU 对比

```bash
SEEKDB_OBS=/path/to/observer \
BASE_F32=/data/base.f32 QUERY_F32=/data/query.f32 GT_I32=/data/gt_100x10.i32 \
DIM=128 N=10000 NQ=100 ROUNDS=3 PORT=2884 OUT_ROOT=/tmp/cpu-gpu \
  bash contest/run_cpu_gpu_compare.sh
```

输出：每轮 `vsag` 与 `cuvs` 的 `metrics.json` 及汇总比值。

## 7. 未验证项与已知风险

| 项 | 状态 | 说明 |
|---|---|---|
| SIFT250K Recall@10/@100 ≥0.95 | **[未验证]** | 无正式数据 |
| Hidden250K Recall@10 ≥0.90 / @100 ≥0.85 | **[未验证]** | 无正式数据 |
| 相对同轮基线召回下降 ≤0.005 | **[未验证]** | 只能在正式数据上测；这是本提交的主要风险 |
| nq=4096 Top-10/Top-100 吞吐 ≥ 基线 90% | **[未验证]** | 代理数据上该比值 0.908–1.327（128d/256d），余量小 |
| 显存 ≤1.25× 基线且 ≤45GB（250K） | **[未验证]** | 10K 上为 1.63–1.68 GB；250K 过程峰值未采样 |
| 主机内存 ≤96GB（250K） | **[未验证]** | 10K 上 1.05–1.12 GB |
| 重启后快照恢复 | **[实测，10K]** | 基线格式；恢复后查询一致、无 CPU 回退 |
| `lib=vsag` 路径不受影响 | **[未验证]** | 改动只在 bridge 的 CAGRA 建图参数，CPU 分支代码未触碰；但 10K 冒烟未单独建过 `lib=vsag` 索引，需在正式环境补一次建表/建索引/查询 |
| GPU 路径真实使用、无静默回退 | **[实测，10K]** | `built=true`、进程持有 renderD fd、无 `lazily built VSAG fallback` |

若主办方在 250K 上测得召回下降超过 0.005，本提交的预期得分为 0；
冻结基线本身即为 4 分保底（分支 `contest-baseline-safe` 只含文档、不含代码改动）。

## 8. 证据与复现入口

| 内容 | 位置 |
|---|---|
| 10K 六轮原始指标/日志 | `contest/evidence/10k-gpu/` |
| 数据生成脚本 | `contest/gen_dev_dataset.py` |
| 单轮基准脚本 | `contest/run_build_bench.sh` |
| CPU/GPU 交替对比脚本 | `contest/run_cpu_gpu_compare.sh` |
| 编译方式 | §2、[`docs/gpu-vector-index-hipvs-cuvs/README.md`](docs/gpu-vector-index-hipvs-cuvs/README.md) |
