# seekdb × AMD hipVS —— CAGRA 索引构建优化：提交说明

本文件是本赛题要求的三项提交产物之一的入口：

1. **实现代码**：`tools/hipvs_bridge/seekdb_cuvs_bridge.c` 的建图参数改动（见 §1）；
2. **可复现的测试方法**：`contest/` 下的脚本 + §3/§4 的计时口径，与赛题 §6 一致；
3. **CPU/GPU 构建性能对比报告**：§6（250K×128 上 CPU `lib=vsag` 与 GPU `lib=cuvs` 的
   建索引、召回、查询吞吐对比，均为本机实测）。

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

### 1.1 改动：NN-Descent 迭代减半 + 中间图度按维度收紧

位置：[`tools/hipvs_bridge/seekdb_cuvs_bridge.c:39`](tools/hipvs_bridge/seekdb_cuvs_bridge.c#L39)

```c
   h->ip->graph_degree = 128;
-  h->ip->intermediate_graph_degree = 256;
-  h->ip->nn_descent_niter = 100;
+  h->ip->intermediate_graph_degree = (dim <= 192) ? 128 : 192;
+  h->ip->nn_descent_niter = 50;
```

基线 CAGRA 建图参数为 `build_algo=NN_DESCENT`、`graph_degree=128`、
`intermediate_graph_degree=256`、`nn_descent_niter=100`。改动只落在建图关键路径上：

- **`nn_descent_niter` 100 → 50**：建图耗时几乎全在 NN-Descent 上——实测 `build_us` 占
  `ddl_wall` 的 94%（42.7/45.4 s，128d）与 89%（44.2/49.8 s，256d），所以迭代次数直接决定
  建图时间；
- **`intermediate_graph_degree` 按维度收紧**：中间图只用于 NN-Descent 候选裁剪，
  不进入最终图，也不影响检索出入口度。`dim≤192` 时 256→128，`dim>192` 时 256→192。
  256 维上更激进的 128 会损失召回（§5.4 探针），192 在召回与提速之间取平衡。

**未改动**的部分：

- 最终图度 `graph_degree=128` —— 决定图体积、检索出入口度与检索代价；
- 检索参数（`itopk_size=max(512, topk*4)`、`search_width=4`）；
- 快照格式、DDL 流程、SQL 语义、CPU `lib=vsag` 路径。

### 1.2 为什么不动 `graph_degree` 与检索参数

建图时间与召回质量共用同一组参数，而赛题对召回（相对同轮基线下降 ≤0.005）与
查询吞吐（≥基线 90%）都是硬门槛，只有建图时间参与计分。因此取舍原则是：
**只在「不影响检索路径」的参数上取收益**。

- 降 `graph_degree`：代理数据上提速与吞吐同时变好（128→64：建图 0.20×、吞吐 1.27–1.66×），
  但召回下降 0.01–0.05，直接吃掉 0.005 的召回预算，且 32 度时代理召回跌到 0.48/0.61；
- 换 `build_algo`（`IVF_PQ` / `AUTO_SELECT`）：改变算法本身，代理召回与建图时间同时劣化
  （同轮 Top-100 吞吐比值 0.73–0.82）；
- 降 `niter` 是唯一"检索路径零变化"的加速项，`intermediate_graph_degree` 同样不进最终图。

`gcdegree` 与 `search_width` 的候选结论保留在 `contest/` 的实验记录中，不进入本轮提交。

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
- 250K 代理集：`python3 contest/gen_sift250k_proxy.py --mode 128|256 ...`（§5.4 的数据来源）；
- 正式数据（SIFT250K 250K×128、Hidden250K 250K×256）由主办方在评测机提供，本地不可得。

### 4.2 单轮基准

```bash
# GPU 路径（lib=cuvs），250K，与 §5.4 的口径一致
SEEKDB_OBS=/path/to/observer BRIDGE_LIB=/path/to/libseekdb_cuvs_bridge.so \
BASE_F32=/data/sift250k/base.f32 QUERY_F32=/data/sift250k/query.f32 \
GT_I32=/data/sift250k/gt10.i32 DIM=128 N=250000 NQ=100 NQ_QPS=4096 TOPK=10 \
PORT=2891 OUT_ROOT=/tmp/e2e \
  bash contest/run_build_bench.sh CAND-128 cuvs

# CPU 路径（lib=vsag），同一 observer 二进制，不传 BRIDGE_LIB
SEEKDB_OBS=/path/to/observer BASE_F32=... GT_I32=... DIM=128 N=250000 PORT=2894 \
  bash contest/run_build_bench.sh CPU-128 vsag
```

`contest/run_build_bench.sh` 每轮使用全新 base-dir 与独立端口，依次完成：
启动 observer → 建表 → 导入 → 计时 DDL 建索引 → 首条查询 → 召回评估
→ `NQ_QPS` 条单行 SQL 的 Top-10/Top-100 吞吐（`gen_query_sql.py` 生成）
→ 采样 VRAM（`rocm-smi`，1s）与 `VmHWM` → 优雅停止 → 写出 `metrics.json`。

同一 observer、不同 bridge 的配对做法：把两版 bridge 放在不同目录，分别用
`BRIDGE_LIB` 指向即可，脚本会用 `loaded-bridge.txt` 校验 `/proc/<pid>/maps`
里加载的确实是预期路径。

### 4.3 配对与重复

只有同一实例、同一轮次的「基线 vs 候选」比值可用。做法：两个版本交替运行 ≥3 轮
（`contest/run_cpu_gpu_compare.sh` 给出交替脚本骨架），比较 `build_us`、`ddl_wall_ms`
的逐轮比值，取几何均值。不同轮次之间的绝对毫秒会漂移（本机观测到同版本轮间
2.1× 波动，见 §5.3），不可跨轮比较绝对值。

### 4.4 查询口径

正式查询为 nq=4096 条单行 SQL，分别 Top-10 / Top-100，统计 QPS 与 Recall。
本提交未改检索代码与 `graph_degree`，检索路径与基线一致；250K 代理集上的实测比值见
§5.4（0.944–1.028），正式数据上的复核结论见 §7。

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

注：§5.1 的 10K 六轮只包含 `nn_descent_niter=50` 这一项（当时 ideg 仍为 256）；
加入 `intermediate_graph_degree` 后的 250K 结果见 §5.4，10K 未复跑——10K 的召回本就饱和在
1.000，无法区分 ideg 的影响。

### 5.2 50K 代理集上的参数方向（探针级，非生产路径）**[实测，代理]**

用链接同一 `libcuvs_c.so` 的独立探针在合成数据上做同轮配对：

| 变体 | build 配对比值 256d / 128d | 代理 Recall@10（256d / 128d） |
|---|---|---|
| 基线（niter=100） | 1.000 / 1.000 | 0.728 / 0.897 |
| niter=50（只改迭代，ideg 保持 256） | 0.601 / 0.565 | 0.841 / 0.954 |
| niter=15 | 0.252 / 0.257 | 1.000 / 0.9997 |

同时用图结构度量（真 top-10 成为邻居的比例、长程边占比）检查图质量，得到与代理召回
**相反**的排序：niter 越大图越接近真 kNN（precision@128 从 0.049 升到 0.755），
但在合成「多簇 + 簇内近重复」数据上代理召回反而下降。结论：**代理召回不能用于判定
niter 方向**，正式召回只能在 250K 数据上测。这也是本提交不采用 niter=15/25 的原因。

### 5.3 环境噪声 **[实测]**

同一版本在不同轮次的绝对耗时相差可达 2.1×（外部 GPU 争用，`rocm-smi` 持续 100%）。
因此 §5.1 只报同轮交替的逐轮比值，不报跨轮绝对值。

### 5.4 250K 规模端到端配对 **[实测，代理数据]**

正式 250K×128（SIFT250K）与 250K×256（Hidden250K）均不开放，故用同位形代理集，
走**完整生产路径**（observer 建表 → `LOAD DATA` → `CREATE VECTOR INDEX` → 首查询 →
SQL 单行查询吞吐），同实例、连续同批次内先跑基线再跑候选，两版之间只换 bridge。

| 维度 | 数据集 | 版本 | DDL 墙钟 | `build_us` | Recall@10 | Top-10 QPS | Top-100 QPS | 显存峰值 |
|---|---|---|---|---|---|---|---|---|
| 128 | SIFT1M 随机 250K 行（seed 20261030，真实 SIFT 几何） | V0 niter=100, ideg=256 | 45.42 s | 42.72 s | 1.000 | 108.97 | 106.88 | 734 MB |
| 128 | 同上 | **本提交** | **21.74 s** | **18.92 s** | **1.000** | **112.00** | **108.31** | **587 MB** |
| 256 | SIFT128 + 噪声 128 维（β=1.0） | V0 niter=100, ideg=256 | 49.81 s | 44.20 s | 0.990 | 93.21 | 90.51 | 862 MB |
| 256 | 同上 | **本提交** | **27.03 s** | **21.63 s** | **0.993** | **87.99** | **86.79** | **730 MB** |

配对结果：

| 指标 | 128d | 256d |
|---|---|---|
| 建图加速（`ddl_wall` 比值） | **2.09×** | **1.84×** |
| 建图加速（`build_us` 比值） | 2.26× | 2.04× |
| 召回变化（Recall@10） | 0.000 | +0.003 |
| 查询吞吐比值（Top-10 / Top-100） | 1.028 / 1.013 | 0.944 / 0.959 |
| 显存峰值比值 | 0.80× | 0.85× |
| GPU 回退事件 | 0 | 0 |
| `graph_bytes` | 256,001,713（不变） | 384,001,713（不变） |

256 维代理集上另用探针（同一 `libcuvs_c.so`、批量 nq=100）测召回逐项差值：相对同轮
基线 Recall@10 −0.0022、Recall@100 −0.0027，均低于 0.005 门槛；代理集的基线召回
（0.991 / 0.968）是本地能构造出的、最接近 Hidden250K（0.985 / 0.914）的工作点。

显存同时下降，是因为 `intermediate_graph_degree` 只影响建图过程的工作图，最终图
`graph_degree=128` 不变，故 `graph_bytes` 与检索路径完全不变——这解释了为什么查询
吞吐比值贴近 1.0 而不是因为"少建了东西"而虚高。

加权的建图加速（0.4×128d + 0.6×256d，按赛题公式取几何加权）：
`exp(0.4·ln2.09 + 0.6·ln1.84) = 1.94×` **[推算]**，对应建图得分
`14·ln(1.94)/ln(3) ≈ 8.4` 分 **[推算]**。是否成立取决于正式数据的召回门槛（§7）。

### 5.5 提交内容自复现 **[实测]**

把本仓库 `tools/hipvs_bridge/` 下的源码与 `contest/run_build_bench.sh` 原样拷到实例上，
从零编译 bridge 后跑 250K×128：`ddl_wall_ms=22641`、`build_us=19.77 s`、
Recall@10 = 1.000、Top-10/Top-100 QPS = 107.41/104.45、`gpu_fallback_events=0`
（证据 `contest/evidence/250k-proxy/REPRO-128/`）。与 §5.4 的 21.74 s 相差 4%
（非同批配对，只作复现性参考；同批建图比值以 §5.4 的 2.09× 为准）。

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

### 6.2 实测数据（250K×128，同实例、同一 observer 二进制） **[实测]**

| 路径 | DDL 墙钟 | 首查询 | Recall@10（GT 精确 top-10） | Top-10 QPS | Top-100 QPS | 显存峰值 | `VmHWM` |
|---|---|---|---|---|---|---|---|
| CPU `lib=vsag`（m=16, ef_construction=200） | 87.34 s | 171 ms | 0.984 | 490.19 | 422.22 | — | 1.49 GB |
| GPU `lib=cuvs`，niter=100（冻结基线） | 45.42 s | 56 ms | 1.000 | 108.97 | 106.88 | 734 MB | 2.67 GB |
| GPU `lib=cuvs`，本提交 | **21.74 s** | 49 ms | 1.000 | 112.00 | 108.31 | 587 MB | 2.18 GB |

同一数据、同一轮次内的比值：

| 比值 | 建索引 | Top-10 吞吐 | Recall@10 |
|---|---|---|---|
| GPU 基线 / CPU | **1.92×** | 0.222 | +0.016 |
| GPU 本提交 / CPU | **4.02×** | 0.228 | +0.016 |
| GPU 本提交 / GPU 基线 | **2.09×** | 1.028 | 0.000 |

读法（都是单轮实测，不是多轮几何均值）：

- 建索引侧 GPU 全面领先：CAGRA 在 250K 规模上比 CPU HNSW 快 1.92×，本提交把这个差距
  拉到 4.02×；
- **nq=1 的单行 SQL 查询上 CPU 反而更快**（490 vs 109 QPS）。原因不在算法而在桥接开销：
  每条查询都要 `cuvsCagraSearchParamsCreate` + 3 次 `cuvsRMMAlloc` + 查询向量 H2D +
  结果 D2H，加上 bridge 侧 `pthread_create(32MB stack)` 与全局锁，固定开销 ~9 ms，
  而 CAGRA 单查询本身只有几十微秒。这是 nq=1 口径的固有开销，**本提交未改动它**，
  因此查询吞吐相对基线基本持平（§5.4 的 0.944–1.028 就是这一段的噪声量级）。
  赛题的查询门槛是「相对同轮基线」而非「相对 CPU」，不受此项影响；
- 召回上 GPU 不低于 CPU（1.000 vs 0.984），GPU 路径没有用召回换速度。

10K×128 的早期数据（基线 6.06/8.14/12.75 s、本提交 3.53/4.23/7.17 s，三对交替）见
§5.1，与 250K 的方向一致，量级差异来自固定开销占比。

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
| SIFT250K Recall@10/@100 ≥0.95 | 代理 **[实测]**，正式 **[未验证]** | 250K SIFT 几何代理上 Recall@10 = 1.000（与基线持平） |
| Hidden250K Recall@10 ≥0.90 / @100 ≥0.85 | 代理 **[实测]**，正式 **[未验证]** | 256 维代理上 0.993（基线 0.990）；代理基线召回 0.991/0.968 已贴近正式工作点 |
| 相对同轮基线召回下降 ≤0.005 | 代理 **[实测]**，正式 **[未验证]** | 探针批量口径 ΔR@10 −0.0022 / ΔR@100 −0.0027（256d）；128d 为 0。这是本提交的主要残余风险，但不再是"未知方向"，只剩"正式数据与代理数据的差距" |
| nq=1 Top-10/Top-100 吞吐 ≥ 基线 90% | 代理 **[实测]**，正式 **[未验证]** | 250K：128d 1.028/1.013，256d 0.944/0.959；图结构与检索参数未变，差异来自轮间噪声 |
| 显存 ≤1.25× 基线且 ≤45GB（250K） | 代理 **[实测]** | 587/734 MB（128d）、730/862 MB（256d），比值 0.80–0.85×，远低于门槛 |
| 主机内存 ≤96GB（250K） | 代理 **[实测]** | `VmHWM` 2.18 GB vs 基线 2.67 GB（128d）；3.17 GB vs 3.27 GB（256d） |
| 重启后快照恢复 | **[实测，10K]** | 基线快照格式未改；恢复后查询一致、无 CPU 回退。250K 上未逐轮验证 |
| `lib=vsag` 路径不受影响 | **[实测，250K 代理]** | 同一 observer 二进制在 250K×128 上建 `lib=vsag` 索引：87.3 s、Recall@10 0.984、无 `lazily built VSAG fallback`（§6.2） |
| GPU 路径真实使用、无静默回退 | **[实测，250K 代理]** | 每轮 `built=true` 且仅一条快照日志、`gpu_fallback_events=0`、进程持有 renderD fd |

剩余风险的量级：赛题召回门槛是「相对同轮基线的下降 ≤0.005」，本提交在 256 维上代理实测
−0.0027（批量 nq=100、Top-100、最差项），余量约 45%。若正式 Hidden250K 的几何分布使
这项超限，本提交在召回门槛上失分；此时可退到 `contest-baseline-safe`（只含文档、不含代码
改动，4 分保底）。

若正式数据上召回通过，建图得分按赛题公式为 `14·clamp(ln s/ln 3)`：
代理加权 `s = exp(0.4·ln2.09 + 0.6·ln1.84) = 1.94` → 8.4 分；查询得分 0（吞吐未超基线
1.0 以上，按门限口径不给分）。合计约 12.4 分 **[推算]**。

## 8. 证据与复现入口

| 内容 | 位置 |
|---|---|
| 10K 六轮原始指标/日志 | `contest/evidence/10k-gpu/` |
| 250K 代理集五轮原始指标/日志 | `contest/evidence/250k-proxy/`（见下表） |
| 数据生成脚本 | `contest/gen_dev_dataset.py` |
| 250K 代理集生成脚本 | `contest/gen_sift250k_proxy.py`（SIFT1M 随机 250K 行；256 维为 SIFT128 + β·噪声128） |
| 单轮基准脚本 | `contest/run_build_bench.sh`（含建索引计时、召回评估、`gen_query_sql.py` 生成的 SQL 单行查询吞吐、内存/显存采样） |
| CPU/GPU 交替对比脚本 | `contest/run_cpu_gpu_compare.sh` |
| 编译方式 | §2、[`docs/gpu-vector-index-hipvs-cuvs/README.md`](docs/gpu-vector-index-hipvs-cuvs/README.md) |

`contest/evidence/250k-proxy/` 下每一轮都含 `metrics.json`（本文所有数字的直接来源）、
`build.log`（`built=true` 的那一条快照日志）、`observer.sha256`/`bridge.sha256`（确认
两侧用的是同一 observer、只是 bridge 不同）、`loaded-bridge.txt`（确认加载的是预期的
bridge）、`vram.samples`（`rocm-smi` 逐秒采样）、`recall.json`、首查询与 Top-10 查询的
返回 id。查询 SQL 与 Top-100 的返回 id 体积大、可用 `contest/gen_query_sql.py` 由
`query.f32` 原样重建，未入库。

| 目录 | 含义 |
|---|---|
| `V0-cuvs` | 250K×128，冻结基线 bridge（niter=100, ideg=256） |
| `CAND-128` | 250K×128，本提交 bridge |
| `V0-256` | 250K×256，冻结基线 bridge |
| `CAND-256` | 250K×256，本提交 bridge |
| `CPU-128` | 250K×128，`lib=vsag`（同一 observer 二进制） |
| `REPRO-128` | 250K×128，由本仓库源码重新编译 bridge、用本仓库脚本复跑（§5.5） |
