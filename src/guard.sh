#!/bin/bash
# Run a command under a resource guard: kill it if the memory footprint of the
# process and its children (as reported by `top`, which - unlike `ps` RSS -
# includes macOS compressed memory) exceeds MAX_MEM_MB, or if free space on the
# working disk drops below MIN_FREE_MB (macOS swap lives on the same disk).
#   usage: MAX_MEM_MB=5000 MIN_FREE_MB=2500 guard.sh <command> [args...]
MAX_MEM_MB=${MAX_MEM_MB:-${MAX_RSS_MB:-5000}}
MIN_FREE_MB=${MIN_FREE_MB:-2500}
mem_mb() {  # summed `top` MEM column (MB) for the given pids
  local args=""
  for p in "$@"; do args="$args -pid $p"; done
  top -l 1 -stats pid,mem $args 2>/dev/null | awk '
    $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9.]+[BKMG]/ {
      v=$2; gsub(/[+-]/, "", v); u=substr(v, length(v)); n=substr(v, 1, length(v)-1)+0
      if (u=="G") n*=1024; else if (u=="K") n/=1024; else if (u=="B") n/=1048576
      s+=n }
    END { printf "%d", s+0 }'
}
"$@" &
PID=$!
while kill -0 $PID 2>/dev/null; do
  MEM=$(mem_mb $PID $(pgrep -P $PID))
  FREE_KB=$(df -k . | awk 'NR==2 {print $4}')
  if [ "$MEM" -gt "$MAX_MEM_MB" ]; then
    echo "GUARD: memory $MEM MB > $MAX_MEM_MB MB - killing $PID" >&2
    pkill -9 -P $PID; kill -9 $PID; exit 137
  fi
  if [ "$FREE_KB" -lt $((MIN_FREE_MB * 1024)) ]; then
    echo "GUARD: free disk $((FREE_KB / 1024)) MB < $MIN_FREE_MB MB - killing $PID" >&2
    pkill -9 -P $PID; kill -9 $PID; exit 137
  fi
  sleep 3
done
wait $PID
