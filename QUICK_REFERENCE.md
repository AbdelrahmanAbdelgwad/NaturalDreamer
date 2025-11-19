# Quick Reference: Testing Commands

## Most Common Use Cases

### 1. Quick Single Test (Baseline)
```bash
python test_checkpoints.py --config cartpole_lyapunov.yml --baseline 100k
```

### 2. Compare Baseline vs Lyapunov
```bash
python test_checkpoints.py --config cartpole_lyapunov.yml \
    --baseline 100k \
    --lyapunov 0.01:100k \
    --num-episodes 20 --test-robustness
```

### 3. Lambda Ablation (Find Best λ)
```bash
python test_checkpoints.py --config cartpole_lyapunov.yml \
    --baseline 100k \
    --lyapunov 0.001:100k 0.01:100k 0.1:100k \
    --num-episodes 15 --test-robustness \
    --output lambda_ablation.csv
```

### 4. Training Progression (How Performance Evolves)
```bash
python test_checkpoints.py --config cartpole_lyapunov.yml \
    --baseline 50k 100k 200k 400k \
    --lyapunov 0.01:50k 0.01:100k 0.01:200k 0.01:400k \
    --num-episodes 10 --output progression.csv
```

### 5. Final Paper Results (Comprehensive)
```bash
# Full comparison with high confidence
python test_checkpoints.py --config cartpole_lyapunov.yml \
    --baseline 100k 200k 400k \
    --lyapunov 0.01:100k 0.01:200k 0.01:400k \
    --num-episodes 30 --test-robustness \
    --output final_results.csv

python visualize_tests.py \
    --results-csv final_results.csv \
    --plot --output-dir final_plots
```

## Command Components

| Component | Example | Description |
|-----------|---------|-------------|
| `--config` | `cartpole_lyapunov.yml` | Config file (auto-finds checkpoints) |
| `--baseline` | `100k 200k` | Space-separated checkpoint steps |
| `--lyapunov` | `0.01:100k 0.1:200k` | Lambda:step pairs |
| `--num-episodes` | `20` | Episodes per evaluation |
| `--test-robustness` | flag | Add noise robustness tests |
| `--output` | `results.csv` | Output filename |

## Visualization

```bash
# With plots
python visualize_tests.py --results-csv results.csv --plot --output-dir plots/

# Summary only
python visualize_tests.py --results-csv results.csv
```

## Tips

- **Quick test**: Use 5-10 episodes
- **Paper results**: Use 20-30 episodes
- **Robustness adds**: ~5min per checkpoint
- **Step format**: "100k" or "100000" both work
- **Output**: Defaults to `test_results.csv`

## Output Files

- `test_results.csv` - All metrics
- `return_comparison.png` - Performance bars
- `failure_rate_comparison.png` - Safety bars
- `action_norm_comparison.png` - Smoothness bars
- `robustness_comparison.png` - Noise curves
