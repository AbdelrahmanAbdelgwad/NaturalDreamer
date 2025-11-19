#!/bin/bash
# Example: Testing baseline vs Lyapunov-regularized checkpoints

# Compare baseline vs Lyapunov at 100k steps
python test_checkpoints.py \
    --config cartpole_lyapunov.yml \
    --baseline 100k \
    --lyapunov 0.01:100k \
    --num-episodes 20 \
    --test-robustness \
    --output results_comparison.csv

# Ablation study: test multiple lambda values
python test_checkpoints.py \
    --config cartpole_lyapunov.yml \
    --baseline 100k \
    --lyapunov 0.001:100k 0.01:100k 0.1:100k \
    --num-episodes 15 \
    --test-robustness \
    --output results_ablation_lambda.csv

# Training progression: compare across multiple checkpoints
python test_checkpoints.py \
    --config cartpole_lyapunov.yml \
    --baseline 50k 100k 200k 400k \
    --lyapunov 0.01:50k 0.01:100k 0.01:200k 0.01:400k \
    --num-episodes 15 \
    --test-robustness \
    --output results_progression.csv

# Quick test at single checkpoint
python test_checkpoints.py \
    --config cartpole_lyapunov.yml \
    --baseline 100k \
    --num-episodes 10 \
    --output results_quick.csv

# Visualize any results
python visualize_tests.py \
    --results-csv results_ablation_lambda.csv \
    --plot \
    --output-dir plots_ablation

# Just print summary without plots
python visualize_tests.py --results-csv results_comparison.csv
