#!/usr/bin/env bash
set -euo pipefail
python -m comm_ai.training.train_neural_bcjr --config src/comm_ai/config/experiments/awgn_small.yaml
