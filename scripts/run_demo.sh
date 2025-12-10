#!/usr/bin/env bash
# Run end-to-end demo: query -> agents -> final -> judge scores
set -euo pipefail

echo "Running end-to-end evaluation demo..."
python main.py --mode evaluate
