#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
"$ROOT_DIR/scripts/experiments/run_config_dir.sh" \
  "$ROOT_DIR/configs/experiments/final/exp_a_noniid_alpha_0_1" "$@"
