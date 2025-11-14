# Stability-Aware DreamerV3 via Latent Lyapunov Regularization

## Overview
This implementation adds Lyapunov stability regularization to the Dreamer algorithm for continuous control tasks. The key idea is to learn a Lyapunov function V(z) that encourages stable and predictable behavior by ensuring monotonic energy decrease along trajectories.

## Key Components

### 1. Lyapunov Function V(z) (`LyapunovModel` in `networks_lyapunov.py`)
- Neural network that maps latent states to scalar energy values
- Ensures positive definiteness: V(z) > 0 for all z ≠ 0
- Trained to correlate with trajectory quality

### 2. Modified Dreamer Agent (`DreamerLyapunov` in `dreamer_lyapunov.py`)
- Includes Lyapunov model alongside actor, critic, and world model
- Modified behavior training with Lyapunov regularization
- Tracks Lyapunov values during evaluation for stability analysis

### 3. Lyapunov Regularization in Actor Loss
The actor loss is modified to include a Lyapunov penalty:
```python
actorLoss = -advantages * logprobs - entropy + λ * lyapunovPenalty
```
Where `lyapunovPenalty` encourages V(z_{t+1}) < V(z_t) along imagined trajectories.

## Files Structure

```
outputs/
├── networks_lyapunov.py      # Network architectures including LyapunovModel
├── dreamer_lyapunov.py       # Modified Dreamer with Lyapunov regularization
├── main_lyapunov.py          # Training script
└── cartpole_lyapunov.yml     # Configuration file with Lyapunov parameters
```

## Key Hyperparameters

In `cartpole_lyapunov.yml`:
- `lyapunovLambda`: Weight for Lyapunov regularization (default: 0.1)
  - Increase for more stable behavior
  - Decrease if stability constraint is too restrictive
- `lyapunovLR`: Learning rate for Lyapunov function (default: 0.0001)
- `lyapunov.hiddenSize`: Network size for V(z) (default: 128)
- `lyapunov.numLayers`: Depth of Lyapunov network (default: 3)

## How to Run

1. **Copy original files to working directory:**
```bash
# Copy utility files that don't need modification
cp buffer.py envs.py utils.py ./
```

2. **Use the Lyapunov versions:**
```bash
# Copy the Lyapunov-enhanced files
cp networks_lyapunov.py networks.py  # Use Lyapunov version of networks
cp dreamer_lyapunov.py dreamer_lyapunov.py
cp main_lyapunov.py main_lyapunov.py
cp cartpole_lyapunov.yml cartpole_lyapunov.yml
```

3. **Train the agent:**
```bash
python main_lyapunov.py --config cartpole_lyapunov.yml
```

## Implementation Details

### Lyapunov Penalty Options
The implementation provides three options for the Lyapunov penalty:

1. **Soft constraint (default):** Penalize when V increases
   ```python
   lyapunovPenalty = torch.relu(lyapunovDifferences).mean()
   ```

2. **Squared penalty:** Smoother gradients
   ```python
   lyapunovPenalty = lyapunovDifferences.pow(2).mean()
   ```

3. **Asymmetric penalty:** Different weights for increases vs decreases
   ```python
   lyapunovPenalty = torch.where(
       lyapunovDifferences > 0,
       lyapunovDifferences * 2,  # Penalize increases
       lyapunovDifferences * 0.1  # Small reward for decreases
   ).mean()
   ```

### Lyapunov Function Training
The Lyapunov function is trained to:
1. Decrease along high-quality trajectories (positive advantages)
2. Increase along poor trajectories (negative advantages)
3. Maintain positive definiteness through squared or exponential activation

### Evaluation Metrics
During evaluation, the implementation tracks:
- Standard performance metrics (reward, episode length)
- Lyapunov monotonicity rate: percentage of steps where V decreases
- Mean and standard deviation of Lyapunov values

## Experimental Tuning

### Start with baseline:
- `lyapunovLambda = 0.1`
- `lyapunovLR = 0.0001`

### If agent is too conservative:
- Decrease `lyapunovLambda` to 0.05 or 0.01
- Check if Lyapunov values are too restrictive

### If agent is unstable:
- Increase `lyapunovLambda` to 0.2 or 0.5
- Consider using squared penalty for smoother optimization

### Monitor during training:
- `lyapunovPenalty`: Should gradually decrease as agent learns stable policies
- `lyapunovMean`: Average Lyapunov value across trajectories
- `monotonicityRate`: Should increase over training (target > 60%)

## Expected Results
- **Performance**: Comparable task reward to baseline Dreamer
- **Stability**: Reduced failure rate and variance in performance
- **Robustness**: Better performance under perturbations
- **Lyapunov monotonicity**: >60% of trajectory steps should show V decrease

## Troubleshooting

1. **Lyapunov values exploding:**
   - Reduce `lyapunovLR`
   - Check gradient clipping is working
   - Consider using exponential instead of squared activation

2. **No stability improvement:**
   - Increase `lyapunovLambda`
   - Check that Lyapunov loss is decreasing
   - Verify Lyapunov values are meaningful (not constant)

3. **Performance degradation:**
   - Decrease `lyapunovLambda`
   - Tune the Lyapunov network architecture
   - Consider using asymmetric penalty

## Citation
Based on the Deep Learning Project Proposal:
"Stability-Aware DreamerV3 via Latent Lyapunov Regularization"
Abdelrahman Abdelgawad, Nana Maryam Munagah, Alexander Choi, Pranav Chintareddy
