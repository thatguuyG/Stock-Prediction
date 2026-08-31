#!/usr/bin/env bash
# Pull fills/positions/equity back from Alpaca into the local DB.
# Safe to run often; it is the only writer to positions and risk_state.
set -euo pipefail

REPO="/home/user/Documents/Stock-Prediction"
LOG_DIR="$REPO/logs"

mkdir -p "$LOG_DIR"
cd "$REPO"
# shellcheck disable=SC1091
source .venv/bin/activate

exec >>"$LOG_DIR/reconcile-$(date +%F).log" 2>&1
echo "--- $(date --iso-8601=seconds)"
stockpred reconcile
