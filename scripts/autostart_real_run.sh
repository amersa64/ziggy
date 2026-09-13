#!/usr/bin/env bash
# Wait for the data hosts to become reachable, then run the real experiment.
#
# The environment's network policy is expected to open at some point. Rather
# than require a human or an agent to be watching at that moment, this polls
# until CONNECT succeeds and then runs the full pipeline unattended.
set -u
LOG=${1:-/tmp/real_run.log}
cd "$(dirname "$0")/.."

probe() {
  curl -sS -o /dev/null -w "%{http_code}" -m 15 \
    -A "ziggy-research amer@zaimler.ai" \
    "https://data.sec.gov/submissions/CIK0000320193.json" 2>/dev/null
}

echo "$(date -u +%H:%M:%S) waiting for data hosts" >> "$LOG"
for _ in $(seq 1 2000); do
  if [ "$(probe)" = "200" ]; then
    echo "$(date -u +%H:%M:%S) HOSTS OPEN - starting real ingestion" >> "$LOG"
    python3 -u -m ziggy.cli all >> "$LOG" 2>&1 && echo "REAL_RUN_DONE" >> "$LOG" \
      || echo "REAL_RUN_FAILED" >> "$LOG"
    exit 0
  fi
  sleep 60
done
echo "$(date -u +%H:%M:%S) gave up waiting for data hosts" >> "$LOG"
echo "REAL_RUN_TIMEOUT" >> "$LOG"
