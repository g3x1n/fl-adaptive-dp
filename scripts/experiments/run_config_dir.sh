#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config_dir> [extra args for run_experiment.py]" >&2
  exit 1
fi

CONFIG_DIR="$1"
shift || true

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="${RUNNER:-scripts/run_experiment.py}"

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Config directory not found: $CONFIG_DIR" >&2
  exit 1
fi

while IFS= read -r config_path; do
  echo "==> Running ${config_path}"
  "$PYTHON_BIN" "$RUNNER" --config "$config_path" "$@"
done < <(
  find "$CONFIG_DIR" -type f -name "*.yaml" \
    | awk '
      {
        order = 2
        if ($0 ~ /\/mnist\//) {
          order = 0
        } else if ($0 ~ /\/cifar10\//) {
          order = 1
        }
        print order "\t" $0
      }
    ' \
    | sort -k1,1n -k2,2 \
    | cut -f2-
)
