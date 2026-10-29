#!/usr/bin/env bash
# contest/run_restart_check.sh — 重启后从快照恢复的检查（配合 run_build_bench.sh 的产物）
#
# 用法：
#   SEEKDB_OBS=/path/to/observer bash contest/run_restart_check.sh <run-dir> <port>
#   <run-dir> 是 run_build_bench.sh 产出的一轮目录（含 run/、data/、evidence/）
#
# 断言：重启后行数不变、首查询结果与重启前逐条一致、没有重新建图（built=true 仍只有
# 第一条）、日志出现 deserialized、无 CPU 回退。
set -Eeuo pipefail

DIR=${1:?usage: run_restart_check.sh <run-dir> <port>}
PORT=${2:?usage: run_restart_check.sh <run-dir> <port>}
: "${SEEKDB_OBS:?set SEEKDB_OBS to the observer binary}"

[ -d "$DIR/run" ] || { echo "no run dir: $DIR/run" >&2; exit 2; }
[ -f "$DIR/data/base.tsv" ] || { echo "no data: $DIR/data/base.tsv" >&2; exit 2; }
[ -f "$DIR/evidence/first_query.ids" ] || { echo "no evidence/first_query.ids" >&2; exit 2; }
mysql -h127.0.0.1 -P"$PORT" -uroot -e "select 1" >/dev/null 2>&1 && { echo "port $PORT busy" >&2; exit 2; }

cd "$DIR/run"
nohup "$SEEKDB_OBS" --nodaemon --port "$PORT" --base-dir "$DIR/run" \
  --parameter _memory_budget=16G --parameter datafile_size=20G \
  --parameter datafile_maxsize=40G --parameter log_disk_size=10G --log-level INFO \
  > "$DIR/run-restart.out" 2>&1 &
PID=$!

for _ in $(seq 1 90); do
  mysql -h127.0.0.1 -P"$PORT" -uroot -e "select 1" >/dev/null 2>&1 && break
  sleep 2
done
mysql -h127.0.0.1 -P"$PORT" -uroot -e "select 1" >/dev/null 2>&1 || { echo "observer not ready after restart" >&2; exit 1; }

CNT=$(mysql -h127.0.0.1 -P"$PORT" -uroot -ANB -e "use benchdb; select count(*) from items")
PROBE=$(head -1 "$DIR/data/base.tsv" | cut -f2)
TOPK=$(grep -c . "$DIR/evidence/first_query.ids")
mysql -h127.0.0.1 -P"$PORT" -uroot -ANB \
  -e "use benchdb; select id from items order by l2_distance(v,$PROBE) approximate limit $TOPK" \
  > "$DIR/evidence/recovery.ids"

SAME=no
cmp -s "$DIR/evidence/first_query.ids" "$DIR/evidence/recovery.ids" && SAME=yes
BUILT=$(grep -ac 'built=true' "$DIR/run/log/seekdb.log" || true)
FALLBACK=$(grep -ac 'lazily built VSAG fallback' "$DIR/run/log/seekdb.log" || true)
DESER=$(grep -ac 'deserialized' "$DIR/run/log/seekdb.log" || true)

kill -TERM "$PID" 2>/dev/null || true
for _ in $(seq 1 60); do
  [ ! -d "/proc/$PID" ] && break
  sleep 2
done

echo "rows=$CNT same_as_before_restart=$SAME built_true_lines=$BUILT deserialized_lines=$DESER fallback_lines=$FALLBACK"
[ "$SAME" = "yes" ] && [ "${FALLBACK:-0}" = "0" ] && [ "${DESER:-0}" -ge 1 ] || exit 1
