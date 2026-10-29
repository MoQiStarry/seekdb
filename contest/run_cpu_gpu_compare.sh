#!/usr/bin/env bash
# contest/run_cpu_gpu_compare.sh — CPU(lib=vsag) 与 GPU(lib=cuvs) 建索引对照
#
# 每轮交替运行两个库（奇偶轮交换先后顺序，降低顺序/温度偏差），
# 最后按轮计算 cuvs/vsag 的墙钟比值并给出几何均值。
#
# 用法（环境变量含义同 run_build_bench.sh）：
#   SEEKDB_OBS=... BASE_F32=... QUERY_F32=... GT_I32=... DIM=128 N=10000 NQ=100 \
#   BRIDGE_LIB=/path/libseekdb_cuvs_bridge.so PORT=2884 \
#   OUT_ROOT=/tmp/cpu-gpu-compare ROUNDS=3 \
#     bash contest/run_cpu_gpu_compare.sh
#
# 注意：2026-10-29 已在 AMD 实例上按同一流程手工跑过一轮 250K×128 的 vsag/cuvs 对照
# （证据 contest/evidence/250k-proxy/CPU-128、CAND-128），本脚本是把它脚本化、多轮化的版本。
set -Eeuo pipefail

ROUNDS=${ROUNDS:-3}
OUT_ROOT=${OUT_ROOT:-/tmp/cpu-gpu-compare}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$OUT_ROOT"

for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then ORDER="cuvs vsag"; else ORDER="vsag cuvs"; fi
  for lib in $ORDER; do
    log_label="r$r-$lib"
    echo "[compare] round $r: $lib"
    bash "$HERE/run_build_bench.sh" "$log_label" "$lib"
  done
done

python3 - "$OUT_ROOT" <<'PY'
import json, math, pathlib, sys

root = pathlib.Path(sys.argv[1])
runs = {}
for p in root.rglob("metrics.json"):
    m = json.loads(p.read_text())
    label = m["label"]
    if not label.startswith("r") or "-" not in label:
        continue
    rnd, lib = label[1:].split("-", 1)
    runs.setdefault(int(rnd), {})[lib] = m

def geo(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")

rows = []
wall_ratios, recall_gap = [], []
for rnd in sorted(runs):
    d = runs[rnd]
    if "cuvs" not in d or "vsag" not in d:
        continue
    c, v = d["cuvs"], d["vsag"]
    wr = v["build_window_ms"] / c["build_window_ms"]
    wall_ratios.append(wr)
    rc = (c.get("recall") or {}).get("recall")
    rv = (v.get("recall") or {}).get("recall")
    if rc is not None and rv is not None:
        recall_gap.append(rc - rv)
    rows.append((rnd, v["build_window_ms"], c["build_window_ms"], wr, c.get("build_us"), rv, rc))

print("| round | vsag window (ms) | cuvs window (ms) | vsag/cuvs | cuvs build_us | vsag recall | cuvs recall |")
print("|---|---|---|---|---|---|---|")
for rnd, vw, cw, wr, bu, rv, rc in rows:
    print("| {} | {} | {} | {:.3f} | {} | {} | {} |".format(
        rnd, vw, cw, wr, bu or "-", "n/a" if rv is None else rv, "n/a" if rc is None else rc))
if wall_ratios:
    g = geo(wall_ratios)
    print("\nbuild-window geometric mean vsag/cuvs = {:.3f}  (cuvs faster by {:.2f}x)".format(g, g))
if recall_gap:
    print("recall delta (cuvs - vsag): {}".format(", ".join("{:+.4f}".format(x) for x in recall_gap)))
PY
