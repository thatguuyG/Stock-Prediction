#!/usr/bin/env bash
# Nightly pipeline: refresh data, score the newest bar, emit signals.
#
# Run AFTER the US close (16:00 ET) so the day's daily bar is final.
# Pass --dry-run as $1 to skip Alpaca submission.
#
# Cron has almost no environment, so everything is absolute and explicit.
set -euo pipefail

REPO="/home/user/Documents/Stock-Prediction"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/daily-$(date +%F).log"
DRY_RUN="${1:-}"

mkdir -p "$LOG_DIR"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

exec >>"$LOG" 2>&1
echo "===== $(date --iso-8601=seconds) starting daily run ====="

# Skip US market holidays cheaply: if no new bar lands, predict/run-signals
# simply re-score the last bar and the upsert keys make it a no-op.
stockpred ingest-prices
stockpred compute-features
stockpred ingest-news
stockpred score-sentiment
stockpred predict --model-version v1
# shellcheck disable=SC2086
stockpred run-signals --model-version v1 $DRY_RUN

echo "===== $(date --iso-8601=seconds) done ====="
