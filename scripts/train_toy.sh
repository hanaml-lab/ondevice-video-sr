#!/usr/bin/env bash
# Train and evaluate a DRIFT student on the synthetic toy world (CPU, < 1 minute).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m drift_vsr.train --config configs/toy.yaml --out toy_metrics.json "$@"
