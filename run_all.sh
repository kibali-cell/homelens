#!/usr/bin/env bash
# Reproduce every number and figure in the study guide. Takes ~10-12 minutes on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")"
rm -f reports/outputs/results.json
for s in steps/step*.py; do python -W ignore tools/run_step.py "$s"; done
python -m pytest -q tests
