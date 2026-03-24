#!/usr/bin/env bash
set -euo pipefail
python -m comm_ai.experiments.run_experiment --config src/comm_ai/config/experiments/awgn_small.yaml
