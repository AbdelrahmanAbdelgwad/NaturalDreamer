import torch
import numpy as np
import argparse
import os
import pandas as pd
import gymnasium as gym
from pathlib import Path
from utils import loadConfig, seedEverything
from envs import CleanGymWrapper, DMControlWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_agent(config, observationShape, actionSize, actionLow, actionHigh, device):
    """Load appropriate agent based on config."""
    if hasattr(config.dreamer, "lyapunov"):
        # Has Lyapunov section - use DreamerLyapunov
        from lyapunov.dreamer_lyapunov import DreamerLyapunov

        return DreamerLyapunov(
            observationShape, actionSize, actionLow, actionHigh, device, config.dreamer
        )
    else:
        # No Lyapunov section - use regular Dreamer
        from dreamer import Dreamer

        return Dreamer(
            observationShape, actionSize, actionLow, actionHigh, device, config.dreamer
        )


class NoisyEnvWrapper(gym.Wrapper):
    """Add Gaussian noise to observations for robustness testing."""

    def __init__(self, env, noise_std=0.0):
        super().__init__(env)
        self.noise_std = noise_std

    def step(self, action):
        obs, reward, done = self.env.step(action)
        if self.noise_std > 0:
            obs = obs + np.random.normal(0, self.noise_std, obs.shape).astype(
                np.float32
            )
        return obs, reward, done

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        if self.noise_std > 0:
            obs = obs + np.random.normal(0, self.noise_std, obs.shape).astype(
                np.float32
            )
        return obs


def find_checkpoint(config, step):
    """Find checkpoint based on config and step."""
    if isinstance(step, str):
        if step.endswith("k"):
            step_num = int(float(step[:-1]) * 1000)
        else:
            step_num = int(step)
    else:
        step_num = int(step)

    step_suffix = f"{step_num/1000:.0f}k"
    checkpoint_dir = Path(config.folderNames.checkpointsFolder)
    env_name = config.environmentName
    run_name = config.runName

    # Primary pattern: {env}_{runName}_{step}
    pattern = f"{env_name}_{run_name}_{step_suffix}*"
    matches = list(checkpoint_dir.glob(pattern))

    if not matches:
        # Fallback: any checkpoint at this step
        pattern2 = f"{env_name}_*{step_suffix}*"
        matches = list(checkpoint_dir.glob(pattern2))

    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found: {checkpoint_dir}/{pattern}\n"
            f"Step: {step}, Run: {run_name}"
        )

    if len(matches) > 1:
        print(f"Warning: Multiple matches for {pattern}:")
        for m in matches:
            print(f"  - {m}")
        matches.sort(key=os.path.getmtime, reverse=True)
        print(f"Using: {matches[0]}")

    return str(matches[0])


def evaluate_checkpoint(agent, env, num_episodes=10, max_steps=1000, seed=42):
    """Evaluate checkpoint and collect metrics."""
    returns = []
    episode_lengths = []
    action_norms = []
    early_terminations = 0

    for ep in range(num_episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        episode_return = 0
        step_count = 0
        episode_actions = []

        h = torch.zeros(1, agent.recurrentSize, device=agent.device)

        while not done and step_count < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                encoded_obs = agent.encoder(obs_tensor)
                z, _ = agent.posteriorNet(torch.cat((h, encoded_obs), -1))
                action = agent.actor(torch.cat((h, z), -1)).cpu().numpy()[0]

            obs, reward, done = env.step(action)
            episode_return += reward
            step_count += 1
            episode_actions.append(action)

            with torch.no_grad():
                action_tensor = (
                    torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
                )
                h = agent.recurrentModel(h, z, action_tensor)

        returns.append(episode_return)
        episode_lengths.append(step_count)

        actions_array = np.array(episode_actions)
        action_norms.append(np.linalg.norm(actions_array, axis=1).mean())

        if step_count < max_steps and done:
            early_terminations += 1

    return {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "mean_episode_length": np.mean(episode_lengths),
        "failure_rate": early_terminations / num_episodes,
        "mean_action_norm": np.mean(action_norms),
        "std_action_norm": np.std(action_norms),
    }


def test_robustness(agent, env, noise_levels, num_episodes=5, seed=42):
    """Test robustness under different noise levels."""
    results = []
    for noise_std in noise_levels:
        noisy_env = NoisyEnvWrapper(env, noise_std=noise_std)
        metrics = evaluate_checkpoint(agent, noisy_env, num_episodes, seed=seed)
        metrics["noise_std"] = noise_std
        results.append(metrics)
    return results


def parse_run_specs(args):
    """Parse run specifications: config:step or config:step:label"""
    specs = []

    if args.baseline:
        for spec in args.baseline:
            if ":" in spec:
                parts = spec.split(":")
                config_file = parts[0]
                step = parts[1]
                label = parts[2] if len(parts) > 2 else f"Baseline-{step}"
            else:
                raise ValueError(f"Baseline must be 'config:step', got: {spec}")
            specs.append((config_file, step, label))

    if args.lyapunov:
        for spec in args.lyapunov:
            if ":" in spec:
                parts = spec.split(":")
                config_file = parts[0]
                step = parts[1]
                label = parts[2] if len(parts) > 2 else f"Lyapunov-{step}"
            else:
                raise ValueError(f"Lyapunov must be 'config:step', got: {spec}")
            specs.append((config_file, step, label))

    return specs


def main(args):
    specs = parse_run_specs(args)

    if not specs:
        print("Error: No runs specified. Use --baseline and/or --lyapunov")
        return

    # Setup environment from first config
    first_config = loadConfig(specs[0][0])
    seedEverything(first_config.seed)

    if "cartpole-swingup" in first_config.environmentName:
        domain, task = first_config.environmentName.split("-")
        env = CleanGymWrapper(DMControlWrapper(domain, task))
    else:
        raise NotImplementedError("Only cartpole-swingup supported")

    observationShape = env.observation_space.shape
    actionSize = env.action_space.shape[0]
    actionLow = env.action_space.low
    actionHigh = env.action_space.high

    all_results = []

    for config_file, step, run_label in specs:
        print(f"\n{'='*60}")
        print(f"Testing: {run_label}")
        print(f"Config: {config_file}, Step: {step}")
        print(f"{'='*60}")

        config = loadConfig(config_file)

        try:
            checkpoint_path = find_checkpoint(config, step)
            print(f"Found: {checkpoint_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

        agent = load_agent(
            config, observationShape, actionSize, actionLow, actionHigh, device
        )
        agent.loadCheckpoint(checkpoint_path)

        print("Running standard evaluation...")
        metrics = evaluate_checkpoint(
            agent, env, num_episodes=args.num_episodes, seed=config.seed
        )
        metrics["run_name"] = run_label
        metrics["config_file"] = config_file
        metrics["step"] = step
        metrics["test_type"] = "standard"
        all_results.append(metrics)

        print(
            f"  Mean Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}"
        )
        print(f"  Failure Rate: {metrics['failure_rate']*100:.1f}%")
        print(f"  Mean Action Norm: {metrics['mean_action_norm']:.3f}")

        if args.test_robustness:
            print("\nRunning robustness tests...")
            noise_levels = [0.0, 0.025, 0.05, 0.075, 0.1]
            robustness_results = test_robustness(
                agent, env, noise_levels, num_episodes=5, seed=config.seed
            )

            for res in robustness_results:
                res["run_name"] = run_label
                res["config_file"] = config_file
                res["step"] = step
                res["test_type"] = "robustness"
                all_results.append(res)
                print(
                    f"  Noise σ={res['noise_std']:.3f}: Return={res['mean_return']:.2f}"
                )

    df = pd.DataFrame(all_results)
    output_path = args.output if args.output else "test_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    print("\nSummary (Standard Evaluation):")
    standard_df = df[df["test_type"] == "standard"]
    summary = standard_df[
        ["run_name", "mean_return", "failure_rate", "mean_action_norm"]
    ]
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test DreamerV3 checkpoints with different configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare baseline vs Lyapunov
  python test_checkpoints.py \\
      --baseline cartpole_baseline.yml:100k \\
      --lyapunov cartpole_lyapunov_0.01.yml:98k

  # With custom labels
  python test_checkpoints.py \\
      --baseline cartpole_baseline.yml:100k:Baseline \\
      --lyapunov cartpole_lyapunov_0.01.yml:98k:λ=0.01 \\
      --lyapunov cartpole_lyapunov_0.1.yml:98k:λ=0.1 \\
      --num-episodes 20 --test-robustness

  # Training progression
  python test_checkpoints.py \\
      --baseline cartpole_baseline.yml:50k:Base-50k \\
      --baseline cartpole_baseline.yml:100k:Base-100k \\
      --lyapunov cartpole_lyapunov.yml:50k:Lyap-50k \\
      --lyapunov cartpole_lyapunov.yml:100k:Lyap-100k
        """,
    )
    parser.add_argument(
        "--baseline", nargs="+", help="Baseline: config:step or config:step:label"
    )
    parser.add_argument(
        "--lyapunov", nargs="+", help="Lyapunov: config:step or config:step:label"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=10, help="Episodes per evaluation"
    )
    parser.add_argument(
        "--test-robustness", action="store_true", help="Test robustness under noise"
    )
    parser.add_argument(
        "--output", type=str, default="test_results.csv", help="Output CSV path"
    )

    args = parser.parse_args()
    main(args)
