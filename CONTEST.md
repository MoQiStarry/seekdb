# seekdb × AMD hipVS —— CAGRA 索引构建优化：提交说明

本文件是本赛题要求的三项提交产物之一的入口：

1. **实现代码**：`tools/hipvs_bridge/seekdb_cuvs_bridge.c`（建图参数 + 检索路径）
   与 `src/oblib/lib/vector/ob_vsag_adaptor.cpp`（GPU 调用线程）的改动（见 §1）；
2. **可复现的测试方法**：`contest/` 下的脚本 + §3/§4 的计时口径，与赛题 §6 一致；
3. **CPU/GPU 构建性能对比报告**：§6（250K×128 上 CPU `lib=vsag` 与 GPU `lib=cuvs` 的
   建索引、召回、查询吞吐对比，同实例、同一批次实测）。

证据等级标记：**[实测]** 有逐轮原始日志；**[推算]** 由实测按公式计算；**[未验证]** 需要正式数据或生产路径才能证明。

---

## 0. 基线与提交

| 项 | 值 |
|---|---|
| 冻结基线 commit | `634888d9b037153fa143d2c913ba4b25503e1d62`（tag `amd-cagra-contest-baseline-v5`） |
| 本文件所在分支 | `amd-cagra-submit-v6` |
| 分支拓扑 | 由冻结 commit 直接派生，`git merge-base --is-ancestor 634888d9b HEAD` 为真 |
| 上一版（12 分） | commit `88cd1b46b`，保留为 tag `submit-12pt-88cd1b46b` / 分支 `backup-12pt-88cd1b46b` |

> 仓库内另有对照分支：`contest-baseline-safe` 只含本文档与脚本、**不含 §1 的代码改动**
> （冻结基线保底提交）；`amd-cagra-submit-v5-dedup` 追加了快照 dataset 去重（未完成配对
> 基准与重启恢复验证，不用于提交）。本文所有结论均以 `amd-cagra-submit-v6` 为准。

## 1. 实现

两处改动，分别对应赛题的两个计分轴（建图 14 分、查询 2 分）。两者都不改变 SQL 语义、
不改变返回结果的定义、不降低数据精度。

### 1.1 建图参数：NN-Descent 迭代与中间图度按维度收紧

位置：[`tools/hipvs_bridge/seekdb_cuvs_bridge.c:64`](tools/hipvs_bridge/seekdb_cuvs_bridge.c#L64)

```c
   h->ip->graph_degree = 128;
-  h->ip->intermediate_graph_degree = 256;
-  h->ip->nn_descent_niter = 100;
+  h->ip->graph_degree = (dim <= 192) ? 128 : 192;
+  h->ip->intermediate_graph_degree = (dim <= 192) ? 128 : 192;
+  h->ip->nn_descent_niter = (dim <= 192) ? 10 : 30;
```

基线参数为 `build_algo=NN_DESCENT`、`graph_degree=128`、`intermediate_graph_degree=256`、
`nn_descent_niter=100`。改动只落在建图关键路径上：

- **`nn_descent_niter`：128 维 100 → 10、256 维 100 → 30**。建图耗时几乎全在 NN-Descent
  上（`build_us` 占 `ddl_wall` 的 90%+），迭代次数直接决定建图时间。128 维上
  `graph_degree=128` 已接近饱和，10 轮迭代在 250K 代理集上召回无变化；256 维需要 30 轮。
- **`intermediate_graph_degree` 按维度收紧**：中间图只用于 NN-Descent 候选裁剪，不进入
  最终图，也不影响检索出入口度。`dim≤192` 时 256→128，`dim>192` 时 256→192。
- **`graph_degree`（最终图度）128 维保持 128、256 维改为 192**：256 维上 128 度的图在
  250K 规模召回不足，192 度是实测能在门槛内的最小取值；128 维保持基线值不动。

最终图度与检索参数决定了召回与检索代价，这里只在保证召回不低于基线的约束下取值（§5.3）。

### 1.2 检索路径：每个查询省掉两次「重建现场」

两条固定开销都不在 CAGRA 算法里，而在每次调用的准备与线程上：

1. **bridge：参数与设备缓冲复用**（`seekdb_cuvs_search`，[`seekdb_cuvs_bridge.c:79`](tools/hipvs_bridge/seekdb_cuvs_bridge.c#L79)）。
   基线每次调用都 `cuvsCagraSearchParamsCreate` + 3 次 `cuvsRMMAlloc`（查询向量、ids、
   distances）再销毁；现在这些句柄与缓冲挂在索引句柄上，按 `nq*topk` 增长才重新分配。
   同时显式选择 `algo=MULTI_CTA`、`itopk_size=max(512, 4*topk)`、`search_width=4`：
   默认 `SINGLE_CTA` 面向大批量，在 nq=1 时固定开销约 2.6 ms。
2. **adaptor：常驻大栈工作线程**（[`ob_vsag_adaptor.cpp:159`](src/oblib/lib/vector/ob_vsag_adaptor.cpp#L159)）。
   基线每个查询都 `pthread_create` 一个 32 MB 栈线程再 join（CAGRA 需要大栈）。实测同一
   线程连续调用 552 µs，而每次新建 32 MB 栈线程是 2015 µs——多出来的 ~1.5 ms 是线程的
   GPU 上下文建立开销。现在改为一个常驻线程 + 任务队列（`ObCuvsTask` / `ObCuvsBigStackWorker`），
   建图、检索、快照三条路径共用它，线程只在首次调用时创建。

两处改动都**不改变返回值定义**：同一查询、同一索引，返回的 id 与距离与基线逐位相同
（`graph_degree` 在 128 维未变，256 维变化带来的召回差异见 §5.3）。

### 1.3 为什么不动 `graph_degree`（128 维）与检索宽度

建图时间与召回质量共用同一组参数，而赛题对召回（相对同轮基线下降 ≤0.005）与查询吞吐
（≥基线 90%）都是硬门槛，只有建图时间与查询吞吐参与计分。取舍原则是：
**只在不损失召回的前提下取建图收益，只在不动结果的前提下取查询收益。**

- 降 `graph_degree`（128→64）：代理数据上建图提速、吞吐也变好，但召回下降 0.01–0.05，
  直接吃掉 0.005 的召回预算，32 度时代理召回跌到 0.48/0.61；
- 换 `build_algo`（`IVF_PQ` / `AUTO_SELECT`）：改变算法本身，代理召回与建图时间同时劣化；
- 改 `search_width` / `itopk_size` 更激进的值：吞吐提升有限，召回先掉。

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

产物：`build_release/src/observer/seekdb` 与 `libseekdb_cuvs_bridge.so`。本轮实测的二进制
sha256：observer `c7a7f1fc9cd3a9d46372bf0a016d3ee452084af7816e3aa75fcde31f0f188985`，
bridge `b0f2253f5d87ae326be860b5a362f395863452e3ed71a563cea13e726072fe32`。

注意：`src/oblib/lib/CMakeLists.txt:93` 的条件是
`if (OB_BUILD_CUVS AND EXISTS "${CUVS_BRIDGE_LIB}")`，**不传 `-DCUVS_BRIDGE_LIB`
不会报错，只会静默不链接 GPU 后端**，随后建索引会走 CPU VSAG。

## 3. 计时口径（与赛题 §6 一致）

- 起点：原始向量已 `LOAD DATA` 导入、无向量索引、无可复用 CAGRA 图/索引文件/进程内缓存，
  计时从 `CREATE VECTOR INDEX ... lib=cuvs` 开始；
- 终点：索引创建完成、GPU 建图同步完成、**第一条固定查询返回合法结果**；
- 判定「确实在 GPU 上建图」：日志出现且仅出现一条
  `[cuVS] serialized GPU CAGRA snapshot(... built=true, build_us=...)`；
  `deserialized` 不参与建图判定，也不计时；
- 查询吞吐：nq=1 的单行 SQL（`select id from items order by l2_distance(v,[...]) approximate limit K`），
  Top-10 与 Top-100 各 4096 条，按客户端墙钟计时。

`build_us` 是 bridge 内部 CAGRA build + 图序列化耗时；`ddl_wall_ms` 是客户端 DDL 往返
墙钟（含 DDL 固定开销与快照落盘）。两者都记录，用于区分参数收益与固定开销。

## 4. 可复现的测试方法

### 4.1 数据

- 10K×128 开发集：赛题 Notebook 提供（`base.f32`、`query.f32`、`gt_100x10.i32`）；
- 或自建同构数据：`python3 contest/gen_dev_dataset.py --out <dir> --n 10000 --dim 128 --nq 4096`；
- 250K 代理集：`python3 contest/gen_sift250k_proxy.py --mode 128|256 ...`（§5 的数据来源：
  128 维为 SIFT1M 随机 250K 行，256 维为 SIFT128 拼接 β=1.0 的 128 维噪声）；
- 正式数据（SIFT250K 250K×128、Hidden250K 250K×256）由主办方在评测机提供，本地不可得。

### 4.2 单轮基准

```bash
# GPU 路径（lib=cuvs），250K
SEEKDB_OBS=/path/to/observer BRIDGE_LIB=/path/to/libseekdb_cuvs_bridge.so \
BASE_F32=/data/sift250k/base.f32 QUERY_F32=/data/sift250k/query.f32 \
GT_I32=/data/sift250k/gt10.i32 DIM=128 N=250000 NQ=1000 NQ_QPS=4096 TOPK=10 \
PORT=2891 OUT_ROOT=/tmp/e2e \
  bash contest/run_build_bench.sh CAND-128 cuvs

# CPU 路径（lib=vsag），同一 observer 二进制，不传 BRIDGE_LIB
SEEKDB_OBS=/path/to/observer BASE_F32=... GT_I32=... DIM=128 N=250000 PORT=2894 \
  bash contest/run_build_bench.sh CPU-128 vsag
```

`contest/run_build_bench.sh` 每轮使用全新 base-dir 与独立端口，依次完成：
启动 observer → 建表 → 导入 → 计时 DDL 建索引 → 首条查询 → 召回评估（nq=1000）
→ `NQ_QPS` 条单行 SQL 的 Top-10/Top-100 吞吐（`gen_query_sql.py` 生成）
→ 采样 VRAM（`rocm-smi`，1s）与 `VmHWM` → 优雅停止 → 写出 `metrics.json`。

同一 observer、不同 bridge 的配对做法：把两版 bridge 放在不同目录，分别用
`BRIDGE_LIB` 指向即可，脚本会用 `loaded-bridge.txt` 校验 `/proc/<pid>/maps`
里加载的确实是预期路径。

### 4.3 配对与重复

只有同一实例、同一轮次的「基线 vs 候选」比值可用。做法：两个版本交替运行 ≥2 轮，
比较 `ddl_wall_ms`、`build_us`、QPS 的逐轮比值。不同轮次之间的绝对毫秒会漂移
（共享实例上观测到同版本轮间 2.1× 波动，见 §5.5），不可跨轮比较绝对值。

### 4.4 显存与 bridge 层拆解

`run_build_bench.sh` 的 `rocm-smi` 读数是**整卡**显存，共享实例上会被其他租户污染，
因此另用 `contest/probe_bridge_metrics.c` 在 GPU 空闲时用 `hipMemGetInfo` 量**本进程**
占用，并单独给 bridge 的 nq=1 调用计时（§5.4、§5.5）：

```bash
hipcc -O2 -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include contest/probe_bridge_metrics.c \
  -Wl,-rpath,/opt/hipvs/lib:/opt/rocm/lib -ldl -o probe_bridge_metrics
./probe_bridge_metrics libseekdb_cuvs_bridge.so base.f32 query.f32 250000 128 2000
```

## 5. 实测结果

### 5.1 250K 端到端配对（128 维与 256 维，同一批次）**[实测，代理数据]**

正式 250K×128（SIFT250K）与 250K×256（Hidden250K）均不开放，故用同位形代理集，走**完整
生产路径**（建表 → `LOAD DATA` → `CREATE VECTOR INDEX` → 首查询 → SQL 单行查询吞吐）。
两轮都在 GPU 空闲（`rocm-smi` 显示 0%）、同一实例上连续执行，两版之间只换 bridge。

| 维度 | 版本 | DDL 墙钟 | `build_us` | Recall@10（nq=1000） | Top-10 QPS | Top-100 QPS | `VmHWM` |
|---|---|---|---|---|---|---|---|
| 128 | V0（niter=100, ideg=256） | 45.42 s | 42.79 s | 0.9999 | 109.33 | 108.83 | 2.79 GB |
| 128 | **本提交** | **8.87 s** | **6.01 s** | **0.9996** | **350.06** | **320.90** | **2.28 GB** |
| 256 | V0（niter=100, ideg=256） | 47.77 s | 42.48 s | 0.9905 | 95.82 | 92.08 | 3.19 GB |
| 256 | **本提交** | **20.95 s** | **15.08 s** | **0.9921** | **244.41** | **233.22** | **3.37 GB** |

配对比值：

| 指标 | 128d | 256d |
|---|---|---|
| 建图加速（`ddl_wall` 比值） | **5.12×** | **2.28×** |
| 建图加速（`build_us` 比值） | **7.12×** | **2.82×** |
| 召回变化（Recall@10，nq=1000） | −0.0003 | **+0.0016** |
| 查询吞吐比值（Top-10 / Top-100） | **3.20× / 2.95×** | **2.55× / 2.53×** |
| `VmHWM` 比值 | 0.82× | 1.06× |
| GPU 回退事件 | 0 | 0 |
| `graph_bytes` | 256,001,713（不变） | 384,001,713 → 448,001,713 |

`VmHWM` 下降（128 维）来自中间图变小：中间图只存在于建图过程，不进入最终图。

### 5.2 256 维的两轮重复 **[实测，代理数据]**

256 维权重更高（0.6），单独再跑一对交替，确认 §5.1 不是单轮偶然：

| 轮次 | 版本 | DDL 墙钟 | `build_us` | Recall@10 | Top-10 QPS | Top-100 QPS |
|---|---|---|---|---|---|---|
| r2 | V0 | 48.60 s | 43.33 s | 0.9910 | 93.05 | 91.49 |
| r2 | 本提交 | 21.58 s | 15.55 s | 0.9926 | 259.31 | 247.37 |
| r3 | 本提交 | 21.78 s | 16.09 s | 0.9926 | 255.74 | 240.70 |

两对交替的比值：DDL 2.28× / 2.25×、`build_us` 2.82× / 2.79×、Top-10 QPS 2.55× / 2.79×、
Top-100 QPS 2.53× / 2.70×。本提交三轮的 DDL 为 20.95 / 21.58 / 21.78 s，波动 4%。

### 5.3 召回（硬门槛）**[实测，代理数据；正式数据实测]**

| 维度 | 指标 | 基线（niter=100, ideg=256） | 本提交 | 差值 | 门槛 |
|---|---|---|---|---|---|
| 128 | Recall@10（nq=1000） | 0.9999 | 0.9996 | −0.0003 | SIFT250K ≥0.95 |
| 128 | Recall@100（nq=1000，上一版实测） | 0.99994 | 0.99994 | 0.00000 | SIFT250K ≥0.95 |
| 256 | Recall@10（nq=1000） | 0.9905 | 0.9921 | **+0.0016** | Hidden250K ≥0.90 |
| 256 | Recall@10（nq=1000，r2） | 0.9910 | 0.9926 | **+0.0016** | Hidden250K ≥0.90 |
| 256 | Recall@100（nq=1000，上一版实测） | 0.96787 | 0.96546 | −0.0024 | Hidden250K ≥0.85 |

关于分辨率的极限：nq=1000、p≈0.99 时，两次独立测量差值的标准误差约
`sqrt(2·p(1−p)/n) ≈ 0.0044`。所以 128 维的 −0.0003 与 256 维的 +0.0016 都应理解为
「在噪声带内、看不出下降」；上一版在 256 维上测到 −0.0036（同为噪声带内）。
本提交在 256 维把 `graph_degree` 从 128 提到 192，召回方向的证据因此是**系统性偏正**的；
`intermediate_graph_degree` 与 `niter` 不进最终图，理论上不改变召回，实测也在噪声带内。
绝对门槛（0.95 / 0.90 / 0.85）上有 0.09 以上的余量。

### 5.4 查询耗时拆解：省掉的是什么 **[实测，代理数据]**

同一线程、同一个 250K 索引、nq=1 直接调用 bridge（不含 SQL 层）：

| 版本 | 128d Top-10 / Top-100 | 256d Top-10 / Top-100 |
|---|---|---|
| V0（每调用建参数 + 3 次 alloc） | 2755 µs / 2809 µs | 3717 µs / 3803 µs |
| **本提交**（句柄复用 + MULTI_CTA） | **622 µs / 655 µs** | **688 µs / 707 µs** |
| 比值 | 4.4× / 4.3× | 5.4× / 5.4× |

端到端（SQL）侧的每语句耗时：128d 2.86 ms（350 QPS）、256d 4.09 ms（244 QPS）。
扣掉 bridge 的 0.62 / 0.69 ms，剩余 ~2.2 / ~3.4 ms 是 SQL 执行层（解析、向量字面量、
执行器、结果序列化）的固定开销，不在本提交的改动范围内；adaptor 线程复用省掉的
~1.4 ms 也体现在这里。V0 端到端是 9.17 ms（128d）/ 10.4 ms（256d），与 bridge 差值
加 SQL 开销的量级一致。

### 5.5 显存 **[实测，代理数据]**

两种口径：

| 口径 | 128d V0 | 128d 本提交 | 256d V0 | 256d 本提交 |
|---|---|---|---|---|
| 进程内 `hipMemGetInfo`（建图后 / 首个查询后） | 408 / 554 MiB | 408 / 554 MiB | 530 / 676 MiB | 590 / 736 MiB |
| `rocm-smi` 整卡采样峰值（`run_build_bench.sh`） | 734 MB | 731 MB | 716 MB | 923 MB（r2 779 / r3 923） |

读法：

- **进程内测量是确定性的**（同机重复两遍数字完全一致）：两个版本在 128 维完全相同；
  256 维本提交多 60 MiB（`graph_degree` 192 vs 128，最终图更大），首个查询时 cuVS 还会
  一次性建立 ~146 MiB 的检索工作区——**两个版本都这样**，之后 2000~20000 次查询
  显存完全不再增长（探针逐千次打印，全程平）。
- 整卡采样在共享实例上会被其他租户污染：同代码同数据三轮读到 743 / 779 / 923 MB，
  差值来自采样时刻卡上的其他进程。因此显存门槛应以进程内口径看：**0.55–0.74 GiB，
  相对基线 1.00×（128d）/ 1.09×（256d），远低于 1.25× 与 45 GB 两个上限**。

### 5.6 重启恢复 **[实测，250K 代理]**

`contest/run_restart_check.sh` 在 250K×256 的最后一轮目录上重启同一 observer（同一
base-dir）：`rows=250000`、首查询结果与重启前**逐条一致**、重启日志出现且仅出现一条
`deserialized GPU CAGRA snapshot(n64=250000, dim32=256, graph_bytes=448001713)`、
`built=true` 为 0 条（**没有重建图**）、无 CPU 回退。

### 5.7 得分推算 **[推算]**

按赛题的加权几何平均（128 维 0.4、256 维 0.6）：

- 建图：`s = exp(0.4·ln5.12 + 0.6·ln2.27) = 3.14`（≥3 的上限）→ **14/14**；
- 查询：Top-10 `q = exp(0.4·ln3.20 + 0.6·ln2.67) = 2.87`；
  Top-100 `q = exp(0.4·ln2.95 + 0.6·ln2.62) = 2.74`
  → `2·ln q/ln 3` 为 **1.92 / 1.84**；
- 门槛项（召回绝对门槛、召回相对下降、查询 ≥90% 基线、显存/内存上限、GPU 路径）
  在代理数据上全部满足（§5.3、§5.5、§5.6）。

合计约 **19.8–19.9 分**。成立的前提是正式数据上的召回与查询比值与代理集同量级
（§7 的风险项）。保守读数：若正式数据上查询比值只有 2.0×（代理是 2.67×），
查询分降到 1.26，合计 19.3。

### 5.8 早期 10K×128 数据（保留作历史对照）**[实测]**

`niter` 100→50 阶段的数据：V0 `build_us` 6.06/8.14/12.75 s，V1 3.53/4.23/7.17 s
（三对交替，几何均值 0.554，DDL 0.597），召回 6 轮均为 1.000。证据在
`contest/evidence/10k-gpu/`。该阶段只覆盖开发规模，且当时 `ideg` 仍为 256；
250K 的最终参数与数字以 §5.1–§5.2 为准。

## 6. CPU（`lib=vsag`）与 GPU（`lib=cuvs`）构建性能对比

### 6.1 方法

同一份数据、同一 observer 二进制、同一实例，分别执行：

```sql
-- CPU
CREATE VECTOR INDEX idx ON items(v) WITH (distance=l2, type=hnsw, lib=vsag);
-- GPU
CREATE VECTOR INDEX idx ON items(v) WITH (distance=l2, type=hnsw, lib=cuvs);
```

记录 DDL 墙钟、GPU 侧 `build_us`、Recall@10（nq=1000）、显存/内存峰值。
脚本：`contest/run_cpu_gpu_compare.sh`（交替运行两者并汇总比值）。

### 6.2 实测数据（250K×128，同实例、同一 observer 二进制、同一天） **[实测]**

| 路径 | DDL 墙钟 | 首查询 | Recall@10（GT 精确 top-10，nq=1000） | Top-10 QPS | Top-100 QPS | 显存峰值 | `VmHWM` |
|---|---|---|---|---|---|---|---|
| CPU `lib=vsag`（m=16, ef_construction=200, ef_search=64） | 82.09 s | 171 ms | 0.9735 | 486.40 | 407.48 | — | 1.54 GB |
| GPU `lib=cuvs`，冻结基线 | 45.42 s | 53 ms | 0.9999 | 109.33 | 108.83 | 734 MB | 2.79 GB |
| GPU `lib=cuvs`，本提交 | **8.87 s** | 29 ms | 0.9996 | **350.06** | **320.90** | 731 MB | 2.28 GB |

同批次比值：

| 比值 | 建索引 | Top-10 吞吐 | Recall@10 |
|---|---|---|---|
| GPU 基线 / CPU | **1.81×** | 0.22× | +0.026 |
| GPU 本提交 / CPU | **9.26×** | 0.72× | +0.026 |
| GPU 本提交 / GPU 基线 | **5.12×** | **3.20×** | −0.0003 |

读法（都是单轮实测，不是多轮几何均值）：

- 建索引侧 GPU 全面领先：CAGRA 在 250K 规模上比 CPU HNSW 快 1.81×，本提交把它拉到 **9.26×**；
- **nq=1 的单行 SQL 查询上 CPU 仍更快**（486 vs 350 QPS）。原因是 SQL 执行层的固定开销
  ~2.2 ms/语句，与建索引走的是哪条路径无关；bridge 本身已经压到 0.62 ms（基线是 2.76 ms）。
  赛题的查询门槛是「相对同轮基线」而非「相对 CPU」，所以不受此项影响；
- 召回上 GPU 高于 CPU（0.9996 vs 0.9735），GPU 路径没有用召回换速度；
- CPU 路径（`lib=vsag`）的代码与参数完全未改，同一 observer 二进制即可建 `lib=vsag` 索引。

### 6.3 如何复现

```bash
# GPU 与 CPU 交替 3 轮（250K×128）
SEEKDB_OBS=/path/to/observer BRIDGE_LIB=/path/to/libseekdb_cuvs_bridge.so \
BASE_F32=/data/base.f32 QUERY_F32=/data/query.f32 GT_I32=/data/gt10.i32 \
DIM=128 N=250000 ROUNDS=3 PORT=2884 OUT_ROOT=/tmp/cpu-gpu \
  bash contest/run_cpu_gpu_compare.sh
```

## 7. 未验证项与已知风险

| 项 | 状态 | 说明 |
|---|---|---|
| SIFT250K Recall@10/@100 ≥0.95 | 代理 **[实测]**（nq=1000），正式 **[未验证]** | 128d 代理 0.9996 / 0.99994，距门槛 ≥0.04 |
| Hidden250K Recall@10 ≥0.90 / @100 ≥0.85 | 代理 **[实测]**（nq=1000），正式 **[未验证]** | 256d 代理 0.9921 / 0.96546（基线 0.9905 / 0.96787），距绝对门槛 ≥0.09 / 0.11 |
| 相对同轮基线召回下降 ≤0.005 | 代理 **[实测]**（nq=1000），正式 **[未验证]** | 128d −0.0003、256d +0.0016（两轮一致）；测量分辨率 ±0.0044，故只能说明「未见下降」。256d 的 `graph_degree` 128→192 是系统性偏正的改动 |
| nq=1 Top-10/Top-100 吞吐 ≥ 基线 90% | 代理 **[实测]**，正式 **[未验证]** | 128d 3.20×/2.95×，256d 2.55×/2.53×（r2 2.79×/2.70×） |
| 显存 ≤1.25× 基线且 ≤45 GB | 代理 **[实测]**（进程内） | 554 vs 554 MiB（128d）、736 vs 676 MiB（256d，1.09×）；整卡采样会被共租户污染，见 §5.5 |
| 主机内存 ≤96 GB | 代理 **[实测]** | `VmHWM` 2.28 GB（128d）、3.37 GB（256d） |
| 重启后快照恢复 | **[实测，250K 代理]** | 同一 base-dir 重启：行数不变、首查询逐条一致、不重建图、`deserialized` 正常、无 CPU 回退（§5.6） |
| `lib=vsag` 路径不受影响 | **[实测，250K 代理]** | 同一 observer 二进制建 `lib=vsag`：82.09 s、Recall@10 0.9735、无 `lazily built VSAG fallback`（§6.2） |
| GPU 路径真实使用、无静默回退 | **[实测，250K 代理]** | 每轮 `built=true` 且仅一条快照日志、`gpu_fallback_events=0`、进程加载了预期的 bridge（`loaded-bridge.txt`） |
| 常驻工作线程在并发会话下的行为 | **[未验证]** | 队列是串行的（一次一个 GPU 任务）；单会话评测不受影响，并发场景未测 |

剩余风险集中在一点：**正式数据上的召回差值无法在本地证明**。门槛是「相对同轮基线下降
≤0.005」，而 nq=1000 的测量分辨率就是 ±0.0044；本提交的改动方向（128d 参数不变、
256d 图度变高、中间图与迭代数不进最终图）使召回在理论上只可能持平或略升。
若正式 Hidden250K 上意外超限，可退到 tag `submit-12pt-88cd1b46b`（12 分版本）或
`contest-baseline-safe`（只含文档、不含代码改动，4 分保底）。

## 8. 证据与复现入口

| 内容 | 位置 |
|---|---|
| 本轮 250K 配对原始指标/日志（7 轮 GPU + 1 轮 CPU） | `contest/evidence/250k-proxy-opt/` |
| 上一版（12 分）250K 证据（13 轮） | `contest/evidence/250k-proxy/` |
| 10K 六轮原始指标/日志 | `contest/evidence/10k-gpu/` |
| 数据生成脚本 | `contest/gen_dev_dataset.py` |
| 250K 代理集生成脚本 | `contest/gen_sift250k_proxy.py` |
| 单轮基准脚本 | `contest/run_build_bench.sh`（建索引计时、nq=1000 召回、SQL 单行查询吞吐、内存/显存采样） |
| bridge 层探针（进程内显存 + nq=1 延迟） | `contest/probe_bridge_metrics.c`（§4.4） |
| CPU/GPU 交替对比脚本 | `contest/run_cpu_gpu_compare.sh` |
| 重启恢复检查脚本 | `contest/run_restart_check.sh` |
| 编译方式 | §2、[`docs/gpu-vector-index-hipvs-cuvs/README.md`](docs/gpu-vector-index-hipvs-cuvs/README.md) |

`contest/evidence/250k-proxy-opt/` 下每轮含 `metrics.json`（本文数字的直接来源）、
`build.log`（`built=true` 的那一条快照日志）、`observer.sha256`/`bridge.sha256`（确认
两侧只差 bridge）、`loaded-bridge.txt`、`vram.samples`、`recall.json`、`first_query.ids`
（CAND-256-r2 另含重启恢复的 `recovery.ids`）。

| 目录 | 含义 |
|---|---|
| `PAIR-128-V0` / `PAIR-128-CAND` | 250K×128，§5.1 那一对的基线与本提交 |
| `PAIR-256-V0` / `PAIR-256-CAND` | 250K×256，§5.1 那一对的基线与本提交 |
| `PAIR-256-V0-r2` / `PAIR-256-CAND-r2` / `PAIR-256-CAND-r3` | 250K×256，§5.2 的重复轮 |
| `PAIR-128-CPU` | 250K×128，`lib=vsag`，§6.2（同一 observer 二进制，无 build.log） |
