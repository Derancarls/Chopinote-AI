#!/usr/bin/env bash
# Watchdog wrapper: auto-restart preprocessing if it crashes
set -o pipefail

cd /root/Chopinote-AI
LOG=/root/autodl-tmp/data/processed/preprocess.log
START_EPOCH=$(date +%s)

# Kill any orphaned workers from a previous crash
cleanup_orphans() {
    local pids
    pids=$(pgrep -f run_fast_preprocess.py 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') | Cleaning $({ echo "$pids" | wc -l; }) orphaned worker(s)..." | tee -a "$LOG"
        echo "$pids" | xargs -r kill 2>/dev/null || true
        sleep 2
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
    fi
}

while true; do
    cleanup_orphans
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Starting preprocessing..." | tee -a "$LOG"
    python scripts/run_fast_preprocess.py 2>&1 | tee -a "$LOG"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        ELAPSED=$(( $(date +%s) - START_EPOCH ))
        echo "$(date '+%Y-%m-%d %H:%M:%S') | Preprocessing completed successfully (total ${ELAPSED}s)." | tee -a "$LOG"
        exit 0
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') | Preprocessing crashed (exit $EXIT_CODE), restarting in 10s..." | tee -a "$LOG"
    sleep 10
done
