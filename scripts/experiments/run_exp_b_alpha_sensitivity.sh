#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
for alpha_dir in \
  "$ROOT_DIR/configs/experiments/final/exp_b_alpha_sensitivity_alpha_1_0" \
  "$ROOT_DIR/configs/experiments/final/exp_b_alpha_sensitivity_alpha_0_5" \
  "$ROOT_DIR/configs/experiments/final/exp_b_alpha_sensitivity_alpha_0_1"; do
  "$ROOT_DIR/scripts/experiments/run_config_dir.sh" "$alpha_dir" "$@"
done
