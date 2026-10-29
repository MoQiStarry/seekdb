#!/usr/bin/env bash
# contest/run_build_bench.sh — 单轮建索引基准（lib=vsag 或 lib=cuvs）
#
# 改编自本队在 AMD 实例上验证过 9 轮的 run_bench_bridge.sh；本仓库内该脚本
# 尚未在任何机器上执行过（开发机无 Linux/ROCm/hipVS）。
#
# 每轮：全新 base-dir + 独占端口 -> 建表 -> LOAD DATA -> 计时 CREATE VECTOR INDEX
#       -> 计时首条查询 -> 召回评估 -> VmHWM/VRAM 采样 -> 优雅停止 -> metrics.json
#
# 用法：
#   SEEKDB_OBS=/path/build_release/src/observer/seekdb \
#   BASE_F32=/data/base.f32 [QUERY_F32=... GT_I32=...] \
#   DIM=128 N=10000 NQ=100 TOPK=10 PORT=2884 OUT_ROOT=/tmp/bench \
#   [BRIDGE_LIB=/path/libseekdb_cuvs_bridge.so] \
#     bash contest/run_build_bench.sh <label> <vsag|cuvs>
#
# 计时口径与赛题一致：构建窗口 = CREATE INDEX 开始 -> 首条固定查询返回。
set -Eeuo pipefail

LABEL=${1:?usage: run_build_bench.sh <label> <vsag|cuvs>}
LIB=${2:?second argument must be vsag or cuvs}
: "${SEEKDB_OBS:?set SEEKDB_OBS to the observer binary}"
: "${BASE_F32:?set BASE_F32 to the base vectors (.f32)}"

DIM=${DIM:-128}
N=${N:-10000}
NQ=${NQ:-100}
TOPK=${TOPK:-10}
PORT=${PORT:-2884}
OUT_ROOT=${OUT_ROOT:-/tmp/seekdb-contest-bench}
BRIDGE_LIB=${BRIDGE_LIB:-}
TS=$(date +%Y%m%d-%H%M%S)
DIR="$OUT_ROOT/$LABEL-$TS"
EV="$DIR/evidence"
DATA="$DIR/data"
mkdir -p "$DIR/run" "$EV" "$DATA"

log()  { echo "[bench:$LABEL] $*"; }
fail() {
  echo "[bench:$LABEL][FAIL] $*" >&2
  [ -f "$DIR/run/log/seekdb.log" ] && cp "$DIR/run/log/seekdb.log" "$EV/seekdb-fail.log" || true
  exit 1
}

case "$LIB" in vsag|cuvs) ;; *) fail "lib must be vsag or cuvs, got '$LIB'";; esac
[ -x "$SEEKDB_OBS" ] || fail "observer not executable: $SEEKDB_OBS"
command -v mysql >/dev/null || fail "mysql client not found"
mysql -h127.0.0.1 -P"$PORT" -uroot -e "select 1" >/dev/null 2>&1 && fail "port $PORT busy"
sha256sum "$SEEKDB_OBS" | awk '{print $1}' > "$EV/observer.sha256"

if [ "$LIB" = "cuvs" ]; then
  [ -n "$BRIDGE_LIB" ] && [ -f "$BRIDGE_LIB" ] || fail "BRIDGE_LIB required for lib=cuvs"
  sha256sum "$BRIDGE_LIB" > "$EV/bridge.sha256"
  export LD_LIBRARY_PATH="$(dirname "$BRIDGE_LIB"):/opt/hipvs/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

python3 - "$BASE_F32" "$DATA/base.tsv" "$DIM" "$N" <<'PY'
import struct, sys
src, dst, dim, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
raw = open(src, "rb").read()
rows = len(raw) // (dim * 4)
if n > 0:
    rows = min(rows, n)
with open(dst, "w") as f:
    for i in range(rows):
        v = struct.unpack_from("<%df" % dim, raw, i * dim * 4)
        f.write("%d\t[%s]\n" % (i, ",".join("%.7f" % x for x in v)))
print("wrote %d rows dim=%d" % (rows, dim))
PY
ROWS=$(wc -l < "$DATA/base.tsv")
[ "$ROWS" = "$N" ] || fail "expected $N rows, converted $ROWS"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export OB_VSAG_TRACE=${OB_VSAG_TRACE:-0}
cd "$DIR/run"
nohup "$SEEKDB_OBS" --nodaemon --port "$PORT" --base-dir "$DIR/run" \
  --parameter _memory_budget=16G --parameter datafile_size=20G \
  --parameter datafile_maxsize=40G --parameter log_disk_size=10G --log-level INFO \
  > "$DIR/run.out" 2>&1 &
PID=$!
echo "$PID" > "$EV/observer.pid"

for _ in $(seq 1 90); do
  mysql -h127.0.0.1 -P"$PORT" -uroot -e "select 1" >/dev/null 2>&1 && break
  sleep 2
done
mysql -h127.0.0.1 -P"$PORT" -uroot -e "select 1" >/dev/null 2>&1 || fail "observer not ready"
if [ "$LIB" = "cuvs" ]; then
  grep -o '/[^ ]*libseekdb_cuvs_bridge\.so' "/proc/$PID/maps" | sort -u > "$EV/loaded-bridge.txt" || true
  grep -qF "$BRIDGE_LIB" "$EV/loaded-bridge.txt" || fail "loaded bridge != $BRIDGE_LIB"
fi

sql() { mysql --local-infile=1 -h127.0.0.1 -P"$PORT" -uroot -ANB -e "$1"; }

sql "create database benchdb; use benchdb; create table items(id bigint primary key, v vector($DIM));"
sql "use benchdb; load data local infile '$DATA/base.tsv' into table items fields terminated by '\t' lines terminated by '\n' (id,v);"
CNT=$(sql "use benchdb; select count(*) from items")
[ "$CNT" = "$N" ] || fail "loaded rows=$CNT, expected $N"

( while true; do
    /opt/rocm/bin/rocm-smi --showmeminfo vram 2>/dev/null \
      | grep -o "Used Memory (B): [0-9]*" | grep -o "[0-9]*"
    sleep 1
  done ) > "$EV/vram.samples" 2>/dev/null &
SAMPLER=$!

T0=$(date +%s%3N)
sql "use benchdb; create vector index idx on items(v) with (distance=l2,type=hnsw,lib=$LIB,m=16,ef_construction=200,ef_search=64);"
T1=$(date +%s%3N)

PROBE=$(head -1 "$DATA/base.tsv" | cut -f2)
sql "use benchdb; select id from items order by l2_distance(v,$PROBE) approximate limit $TOPK" > "$EV/first_query.ids"
T2=$(date +%s%3N)

kill "$SAMPLER" 2>/dev/null || true
WAIT=0; while kill -0 "$SAMPLER" 2>/dev/null && [ "$WAIT" -lt 3 ]; do sleep 1; WAIT=$((WAIT+1)); done
VRAM_PEAK=$(sort -n "$EV/vram.samples" 2>/dev/null | tail -1 || true)
FIRST_N=$(grep -c . "$EV/first_query.ids" || true)
[ "${FIRST_N:-0}" = "$TOPK" ] || fail "first query returned ${FIRST_N:-0} ids, expected $TOPK"
DUP=$(sort "$EV/first_query.ids" | uniq -d | wc -l)
[ "${DUP:-0}" = "0" ] || fail "first query returned duplicate ids"

BUILD_US=0
GRAPH_BYTES=0
if [ "$LIB" = "cuvs" ]; then
  grep -a '\[cuVS\] serialized GPU CAGRA snapshot' "$DIR/run/log/seekdb.log" \
    | grep -a 'built=true' > "$EV/build.log" || true
  NB=$(wc -l < "$EV/build.log")
  [ "$NB" = "1" ] || fail "expected exactly 1 fresh GPU build, got $NB"
  BUILD_US=$(grep -oa 'build_us=[0-9]*' "$EV/build.log" | cut -d= -f2)
  GRAPH_BYTES=$(grep -oa 'graph_bytes=[0-9]*' "$EV/build.log" | cut -d= -f2)
fi

FALLBACK=$(grep -ac 'lazily built VSAG fallback' "$DIR/run/log/seekdb.log" || true)

RECALL_JSON='{"recall": null}'
if [ -n "${QUERY_F32:-}" ] && [ -n "${GT_I32:-}" ]; then
  python3 - "$PORT" "$QUERY_F32" "$GT_I32" "$NQ" "$TOPK" "$DIM" "$EV/recall.json" <<'PY'
import json, struct, subprocess, sys
port, qf, gf, nq, topk, dim, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
q = open(qf, "rb").read()
gt_raw = open(gf, "rb").read()
gt = struct.unpack("<%di" % (len(gt_raw) // 4), gt_raw)
nq_avail = len(q) // (dim * 4)
nq = min(nq, nq_avail)
stride = len(gt) // nq_avail if nq_avail else 0
if stride < topk:
    raise SystemExit("gt file has %d ids per query, need >= %d" % (stride, topk))
hits = total = 0
for i in range(nq):
    v = struct.unpack_from("<%df" % dim, q, i * dim * 4)
    lit = "[" + ",".join("%.7f" % x for x in v) + "]"
    out_ids = subprocess.run(
        ["mysql", "-h127.0.0.1", "-P" + port, "-uroot", "-ANB", "-e",
         "use benchdb; select id from items order by l2_distance(v,%s) approximate limit %d" % (lit, topk)],
        capture_output=True, text=True).stdout.split()
    got = set(int(x) for x in out_ids)
    hits += len(got & set(gt[i * stride:i * stride + topk]))
    total += topk
json.dump({"nq": nq, "topk": topk, "recall": round(hits / total, 6) if total else None,
           "stride": stride}, open(out, "w"), indent=1)
print(open(out).read())
PY
  RECALL_JSON=$(cat "$EV/recall.json")
fi

VMHWM=$(grep -o "VmHWM:[[:space:]]*[0-9]*" "/proc/$PID/status" | grep -o "[0-9]*" || true)

kill -TERM "$PID" 2>/dev/null || true
for _ in $(seq 1 90); do
  if [ ! -d "/proc/$PID" ] || [ "$(awk '/^State:/{print $2}' "/proc/$PID/status" 2>/dev/null)" = "Z" ]; then break; fi
  sleep 2
done
mysql -h127.0.0.1 -P"$PORT" -uroot -e "select 1" >/dev/null 2>&1 && fail "port not released"

cp "$DIR/run/log/seekdb.log" "$EV/seekdb.log" 2>/dev/null || true
cat > "$EV/metrics.json" <<EOF
{
  "label": "$LABEL",
  "lib": "$LIB",
  "ts": "$TS",
  "observer_sha256": "$(cat "$EV/observer.sha256")",
  "bridge_sha256": "$(cut -d' ' -f1 "$EV/bridge.sha256" 2>/dev/null || echo "")",
  "build_us": ${BUILD_US:-0},
  "graph_bytes": ${GRAPH_BYTES:-0},
  "ddl_wall_ms": $((T1 - T0)),
  "first_query_ms": $((T2 - T1)),
  "build_window_ms": $((T2 - T0)),
  "gpu_fallback_events": ${FALLBACK:-0},
  "recall": $RECALL_JSON,
  "vram_peak_bytes": ${VRAM_PEAK:-0},
  "vmhwm_kb": ${VMHWM:-0},
  "n": $N, "dim": $DIM, "nq": $NQ, "topk": $TOPK, "port": $PORT, "dir": "$DIR"
}
EOF
log "DONE lib=$LIB build_us=${BUILD_US:-0} window_ms=$((T2 - T0)) ddl_ms=$((T1 - T0)) first_query_ms=$((T2 - T1)) fallback=$FALLBACK"
log "metrics: $EV/metrics.json"
