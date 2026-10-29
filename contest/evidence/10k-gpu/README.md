# 10K×128 建索引证据（开发规模）

来源：2026-10-29，AMD Radeon 48GB + ROCm 7.x 实例，同一时段 V0/V1 交替 3 轮
（V0 = 冻结基线 niter=100，V1 = niter=50），数据为赛题提供的 10K×128 开发集。
每个子目录是一次独立运行，`run_build_bench.sh` 的同类输出。

| 文件 | 内容 |
|---|---|
| `metrics.json` | `build_us`、`ddl_wall_ms`、`graph_bytes`、`recall_at_10`、`vram_peak_bytes`、`vmhwm_kb` |
| `build.log` | 唯一一条 `[cuVS] serialized GPU CAGRA snapshot(... built=true ...)` |
| `recall10.json` | 100 条查询的 Recall@10 及其墙钟 |
| `bridge.sha256` | 该轮加载的 bridge 动态库校验值（V0=`030b3624…`，V1=`21c78f83…`） |
| `vram.samples` | 建索引期间 1s 采样的设备显存占用 |

判读注意：不同轮次之间绝对耗时漂移可达 2.1×（外部 GPU 争用），
只有同轮 V0/V1 比值可用；详见仓库根目录 `CONTEST.md` §5。
