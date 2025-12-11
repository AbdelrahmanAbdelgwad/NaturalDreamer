# README: Using the Lyapunov-Augmented NaturalDreamer Repository

This repository extends **NaturalDreamer** (a clean PyTorch DreamerV3 implementation) with additional modules for:

- Lyapunov-regularized policy learning  
- DMControl environment support  
- Visualization of learned dynamics, world-model predictions, and Lyapunov function landscapes  

This document explains **how to use every script**, **what each component does**, and **how to train, evaluate, visualize, and debug** the Lyapunov-enhanced Dreamer agent.

---

# 1. Installation

```bash
git clone https://github.com/AbdelrahmanAbdelgwad/NaturalDreamer.git
cd NaturalDreamer
pip install -r requirements.txt
```

Install DMControl manually:

```bash
pip install dm_control
```

---

# 2. Repository Structure (Practical View)

```
├── main.py                      # Baseline NaturalDreamer training
├── main_lyapunov.py             # Lyapunov-regularized Dreamer training
│
├── dreamer.py                   # Baseline Dreamer agent
├── lyapunov/
│   ├── dreamer_lyapunov.py      # Lyapunov-extended Dreamer agent
│   └── networks_lyapunov.py     # Neural modules incl. LyapunovModel
│
├── envs.py                      # Gym + DMControl environment wrappers
│
├── render_dynamics.py           # Compare real env vs world-model predictions
├── render_world_model.py        # Compare real vs imagined trajectories
│
├── plot_lyapunov.py             # Visualize learned Lyapunov function
│
├── test_checkpoints.py          # Evaluate baseline vs Lyapunov checkpoints
├── visualize_tests.py           # Plot evaluation CSVs
│
├── configs/                     # YAML configuration files
├── utils.py                     # Logging, configs, checkpoint helpers
└── buffer.py                    # Replay buffer implementation
```

---

# 3. Training the Agents

## 3.1 Train Baseline Dreamer (No Lyapunov)

```bash
python main.py --config cartpole_lyapunov.yml
```

Runs DreamerV3 with DMControl Cartpole-Swingup and saves checkpoints.

---

## 3.2 Train Lyapunov-Regularized Dreamer

```bash
python main_lyapunov.py --config cartpole_lyapunov.yml
```

Adds:
- Lyapunov value computation V(z,h)
- Actor penalty for increases in V
- Lyapunov model training
- Stability metrics logging

Key config fields:
- `lyapunovLambda`
- `lyapunovLR`
- `equilibriumPoint`

---

# 4. Environment Wrappers

Provided in `envs.py`:

- `DMControlWrapper(domain, task)`
- `CleanGymWrapper`
- `GymPixelsProcessingWrapper`

Automatically invoked by training scripts.

---

# 5. Visualizing the Learned Lyapunov Function

```bash
python plot_lyapunov.py --checkpoint checkpoints/cartpole_xxx.pth                         --config cartpole_lyapunov.yml                         --output-dir plots/
```

Produces:
- V(θ, θdot) surfaces  
- V(x, xdot) surfaces  
- V(x, θ) surfaces  
- PCA-based latent-space slices  

Useful for diagnosing positivity, shape, and decrease conditions.

---

# 6. Visualizing Dynamics

## 6.1 Real vs World-Model Dynamics

```bash
python render_dynamics.py --config cartpole_lyapunov.yml                           --controller actor                           --horizon 500                           --checkpoint_suffix 100k
```

Outputs:
- Real trajectory
- World-model rollout under same actions
- Per-state plots (vector obs)
- Side-by-side video (pixel obs)

---

## 6.2 Real vs Imagined Trajectories

```bash
python render_world_model.py       --config cartpole_lyapunov.yml       --checkpoint checkpoints/cartpole_xxx.pth       --steps 200 --warmup 50 --output comparison.mp4
```

Outputs:
- Real vs imagined frames side-by-side
- Anchored imagination every N steps

Useful for diagnosing world-model drift.

---

# 7. Evaluating Checkpoints

Quick evaluation:

```bash
python test_checkpoints.py --config cartpole_lyapunov.yml --baseline 100k
```

Compare baseline vs Lyapunov:

```bash
python test_checkpoints.py --config cartpole_lyapunov.yml     --baseline 100k     --lyapunov 0.01:100k     --num-episodes 20     --test-robustness
```

Outputs:
- Average return
- Failure statistics
- Smoothness metrics
- Robustness curves
- CSV results

---

# 8. Plotting Test Results

```bash
python visualize_tests.py --results-csv results.csv --plot --output-dir plots/
```

Generates:
- Return bars  
- Failure rates  
- Action norms  
- Noise curves  

---

# 9. Working With Checkpoints

Saved under:

```
checkpoints/<env>_<runname>_<step>.pth
```

Includes:
- Encoder/decoder
- Recurrent, prior, posterior
- Reward model
- (Optional) continuation
- Actor/critic
- LyapunovModel (Lyapunov version)
- Optimizer states

Load manually:

```python
from lyapunov.dreamer_lyapunov import DreamerLyapunov
agent = DreamerLyapunov(obs_shape, action_size, low, high, device, config)
agent.loadCheckpoint("path/to/file.pth")
```

---

# 10. Recommended Workflow

1. Train agent  
2. Visualize world-model (`render_dynamics.py`)  
3. Visualize imagination (`render_world_model.py`)  
4. Evaluate checkpoints (`test_checkpoints.py`)  
5. Plot results (`visualize_tests.py`)  
6. Inspect Lyapunov function shape (`plot_lyapunov.py`)  

---

# 11. Notes

- Scripts originated during iterative development; this README documents only the functional, intended usage paths.
- GPU recommended for all visualization tools.
- DMControl installation required for DMControl experiments.

---

# 12. Contact

**Abdelrahman Abdelgawad**  
Email: aaoaa@bu.edu

