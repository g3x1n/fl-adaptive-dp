#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
"$ROOT_DIR/scripts/experiments/run_config_dir.sh" \
  "$ROOT_DIR/configs/experiments/final/exp_d_privacy_tradeoff" "$@"
