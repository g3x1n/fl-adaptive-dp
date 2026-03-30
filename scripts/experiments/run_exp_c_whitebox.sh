#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ATTACK_SCRIPT="${ATTACK_SCRIPT:-scripts/run_whitebox_attack.py}"
CONFIG_ROOT="$ROOT_DIR/configs/experiments/final/exp_d_privacy_tradeoff"

for dataset in mnist cifar10; do
  for algorithm in fedavg fedprox fednova; do
    baseline_config="$CONFIG_ROOT/$dataset/${algorithm}.yaml"
    fixed_dp_config="$CONFIG_ROOT/$dataset/${algorithm}_fixed_dp.yaml"
    adaptive_dp_config="$CONFIG_ROOT/$dataset/${algorithm}_adaptive_dp.yaml"

    echo "==> White-box attack: ${dataset} ${algorithm} vs fixed DP"
    "$PYTHON_BIN" "$ROOT_DIR/$ATTACK_SCRIPT" \
      --baseline-config "$baseline_config" \
      --dp-config "$fixed_dp_config" "$@"

    echo "==> White-box attack: ${dataset} ${algorithm} vs adaptive DP"
    "$PYTHON_BIN" "$ROOT_DIR/$ATTACK_SCRIPT" \
      --baseline-config "$baseline_config" \
      --dp-config "$adaptive_dp_config" "$@"
  done
done
