# Checkpoint Testing Scripts

Scripts for evaluating DreamerV3 checkpoints with and without Lyapunov regularization.

## Quick Start

```bash
# Compare baseline vs Lyapunov at 100k steps
python test_checkpoints.py --config cartpole_lyapunov.yml \
    --baseline 100k \
    --lyapunov 0.01:100k \
    --num-episodes 20 --test-robustness

# Visualize results
python visualize_tests.py --results-csv test_results.csv --plot
```

## Files

- `test_checkpoints.py`: Main testing script
- `visualize_tests.py`: Visualization and summary generation
- `example_testing.sh`: Example usage commands

## Checkpoint Specification

**Baseline runs:**
```bash
--baseline 100k 200k 400k
```

**Lyapunov runs (specify lambda:step):**
```bash
--lyapunov 0.01:100k 0.1:100k
```

The script automatically finds checkpoints based on your config's checkpoint directory and naming convention.

## Metrics Collected

**Standard Evaluation:**
- Mean episode return (± std)
- Failure rate (% early terminations)
- Mean action norm (behavioral smoothness)
- Episode length statistics

**Robustness Testing:**
- Performance degradation under Gaussian observation noise (σ ∈ [0, 0.1])

## Usage Examples

### 1. Quick Test Single Checkpoint

```bash
python test_checkpoints.py \
    --config cartpole_lyapunov.yml \
    --baseline 100k \
    --num-episodes 10
```

### 2. Compare Baseline vs Lyapunov

```bash
python test_checkpoints.py \
    --config cartpole_lyapunov.yml \
    --baseline 100k \
    --lyapunov 0.01:100k \
    --num-episodes 20 \
    --test-robustness \
    --output comparison.csv
```

### 3. Lambda Ablation Study

```bash
python test_checkpoints.py \
    --config cartpole_lyapunov.yml \
    --baseline 100k \
    --lyapunov 0.001:100k 0.01:100k 0.1:100k \
    --num-episodes 15 \
    --test-robustness \
    --output ablation.csv
```

### 4. Training Progression

```bash
python test_checkpoints.py \
    --config cartpole_lyapunov.yml \
    --baseline 50k 100k 200k 400k \
    --lyapunov 0.01:50k 0.01:100k 0.01:200k 0.01:400k \
    --num-episodes 15 \
    --output progression.csv
```

### 5. Visualize Results

```bash
# Generate plots and print summary
python visualize_tests.py \
    --results-csv ablation.csv \
    --plot \
    --output-dir plots/

# Just print summary table
python visualize_tests.py --results-csv comparison.csv
```

## Output

**CSV Format:**
- `run_name`: Readable run identifier (e.g., "Baseline-100k", "Lyapunov-λ0.01-100k")
- `run_type`: 'baseline' or 'lyapunov'
- `lambda`: Lambda value (for Lyapunov runs)
- `step`: Training step
- `test_type`: 'standard' or 'robustness'
- `mean_return`: Average episode return
- `std_return`: Standard deviation of returns
- `failure_rate`: Fraction of episodes with early termination
- `mean_action_norm`: Average L2 norm of actions
- `noise_std`: Noise level (for robustness tests)

**Generated Plots:**
- `return_comparison.png`: Bar chart of returns
- `failure_rate_comparison.png`: Bar chart of failure rates
- `action_norm_comparison.png`: Action norm comparison
- `robustness_comparison.png`: Return vs noise level curves

## Checkpoint Naming Convention

The script expects checkpoints to follow these naming patterns:

**Baseline:**
- `{env}_*Baseline*_{step}k`
- Example: `cartpole-swingup_CartpoleSwingup-Baseline_100k`

**Lyapunov:**
- `{env}_*Lyapunov*Lambda{λ}*_{step}k`
- Example: `cartpole-swingup_CartpoleSwingup-Lyapunov-Lambda0.01_100k`

The script will search your config's `checkpointsFolder` for matching patterns.

## Notes

- Higher `num-episodes` gives more reliable statistics but takes longer
- Robustness testing runs 5 episodes per noise level (5 levels total)
- Uses same seed as training config for reproducibility
- Script automatically handles checkpoint path resolution
