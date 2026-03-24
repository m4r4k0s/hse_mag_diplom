#!/usr/bin/env bash
set -euo pipefail
python -m comm_ai.training.train_neural_viterbi --config src/comm_ai/config/experiments/awgn_small.yaml
