# Stability-Aware DreamerV3 via Latent Lyapunov Regularization

**Course Project**: EC523 Deep Learning, Fall 2025, Boston University

**Team**: Abdelrahman Abdelgawad, Nana Maryam Munagah, Alexander Choi, Pranav Chintareddy

## Overview

This project explores integrating Lyapunov stability theory into model-based reinforcement learning. We extend DreamerV3 with a learned Lyapunov function V(z,h) that acts as a soft regularizer, encouraging policies to exhibit stable, dissipative behaviors in latent space while maximizing task reward.

**Research Question**: Can Lyapunov regularization improve controller robustness and stability without sacrificing task performance?

## Key Modifications

Starting from the [NaturalDreamer](https://github.com/InexperiencedMe/NaturalDreamer) implementation, we added:

1. **Lyapunov Head**: 2-layer MLP that estimates energy V(z,h) in latent space
2. **Actor Regularization**: Modified actor loss to penalize increases in V along imagined trajectories
3. **Stability Metrics**: Logging for Lyapunov values, monotonicity rates, and trajectory differences
4. **Environment Adaptation**: Extended codebase to support DMControl Cartpole-Swingup

## Repository Structure
```
├── lyapunov/
│   ├── dreamer_lyapunov.py       # Main Dreamer agent with Lyapunov extension
│   └── networks_lyapunov.py      # Neural network architectures including LyapunovModel
├── configs/
│   └── cartpole_lyapunov.yml     # Hyperparameters for Cartpole experiments
├── main_lyapunov.py               # Training script
├── envs.py                        # Environment wrappers
└── utils.py                       # Utilities for logging, plotting, checkpointing
```

## Installation
```bash
# Clone repository
git clone https://github.com/AbdelrahmanAbdelgwad/NaturalDreamer.git
cd NaturalDreamer

# Install dependencies
pip install torch torchvision gymnasium dm_control pyyaml matplotlib
```

## Usage

### Training with Lyapunov Regularization
```bash
python main_lyapunov.py --config cartpole_lyapunov.yml
```

### Key Hyperparameters

In `cartpole_lyapunov.yml`:
- `lyapunovLambda`: Weight for Lyapunov regularization (default: 0.1)
- `lyapunovLR`: Learning rate for Lyapunov function (default: 0.0001)
- `equilibriumPoint`: Target equilibrium state for stability (default: upright position)

### Training Baseline (No Lyapunov)

Set `lyapunovLambda: 0.0` in config or use the original NaturalDreamer training script.

## Preliminary Results

**Environment**: DMControl Cartpole-Swingup

| Version | Peak Return (100k steps) | Notes |
|---------|-------------------------|-------|
| Baseline DreamerV3 | ~700 | Stable learning |
| Lyapunov λ=0.1 | ~195 | Severe degradation - under investigation |

Current work focuses on diagnosing the performance gap through hyperparameter sweeps and optimization analysis.

## Acknowledgments

This project adapts the [NaturalDreamer](https://github.com/InexperiencedMe/NaturalDreamer) implementation, which provides a clean, modular DreamerV3 implementation. We thank the original authors for making their code available.

**Original DreamerV3 Paper**:  
Hafner et al., "Mastering Diverse Domains through World Models", *Nature*, 2025.

**Lyapunov RL Reference**:  
Chow et al., "A Lyapunov-based Approach to Safe Reinforcement Learning", *NeurIPS*, 2018.

## License

This project inherits the license from the original NaturalDreamer repository.

## Contact

For questions about this project, contact: aaoaa@bu.edu