import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import gymnasium as gym

from dreamer import Dreamer
from envs import GymPixelsProcessingWrapper, CleanGymWrapper, getEnvProperties
from utils import loadConfig, seedEverything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dreamer on CarRacing-v3")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="YAML config used for training (e.g. configs/car-racing-v3.yml)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to Dreamer checkpoint (.pth)",
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=-1,
        help="Training step of this checkpoint (for logging only)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=1000,
        help="Safety cap on episode length",
    )
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to actions (for robustness)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for eval",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="eval_results/eval_carracing.csv",
        help="CSV file to save/append results",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="baseline",
        help="Label for this agent (for plotting)",
    )

    return parser.parse_args()


def set_global_seed(seed: int):
    # Reuse your existing helper + numpy + torch seeds
    seedEverything(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_env_and_dreamer(config_file: str):
    """
    Build CarRacing env + Dreamer exactly like training.
    """
    cfg = loadConfig(config_file)

    if "CarRacing" not in cfg.environmentName:
        raise ValueError(
            f"Expected a CarRacing environment, got {cfg.environmentName}"
        )

    # This mirrors the CarRacing branch in main.py
    env = CleanGymWrapper(
        GymPixelsProcessingWrapper(
            gym.wrappers.ResizeObservation(
                gym.make(cfg.environmentName, render_mode="rgb_array"), (64, 64)
            )
        )
    )

    observationShape, actionSize, actionLow, actionHigh = getEnvProperties(env)
    print(
        f"[Eval] envProperties: obs {observationShape}, "
        f"action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}"
    )

    dreamer = Dreamer(
        observationShape, actionSize, actionLow, actionHigh, device, cfg.dreamer
    )

    return env, dreamer, cfg.environmentName


def run_eval_carracing(
    config_file: str,
    checkpoint_path: str,
    checkpoint_step: int,
    num_episodes: int,
    max_episode_steps: int,
    noise_sigma: float,
    seed: int,
    output_csv: str,
    agent_type: str,
):
    set_global_seed(seed)

    env, dreamer, env_name = build_env_and_dreamer(config_file)

    # Load Dreamer weights
    print(f"[Eval] Loading checkpoint from {checkpoint_path}")
    dreamer.loadCheckpoint(checkpoint_path)

    dreamer.encoder.eval()
    dreamer.actor.eval()
    dreamer.recurrentModel.eval()
    dreamer.posteriorNet.eval()

    rng = np.random.RandomState(seed)

    results = []

    for ep in range(num_episodes):
        # CleanGymWrapper reset gives just obs
        obs = env.reset(seed=seed + ep)
        currentScore = 0.0
        stepCount = 0
        done = False

        # Initialize Dreamer recurrent state
        recurrentState = torch.zeros(1, dreamer.recurrentSize, device=device)
        latentState = torch.zeros(1, dreamer.latentSize, device=device)
        action = torch.zeros(1, dreamer.actionSize, device=device)

        # Encode initial observation
        obs_tensor = torch.from_numpy(np.asarray(obs)).float().unsqueeze(0).to(device)
        encodedObservation = dreamer.encoder(
            obs_tensor.view(-1, *dreamer.observationShape)
        )

        action_norms = []

        while not done and stepCount < max_episode_steps:
            # World-model update (same pattern as environmentInteraction)
            recurrentState = dreamer.recurrentModel(recurrentState, latentState, action)
            latentState, _ = dreamer.posteriorNet(
                torch.cat((recurrentState, encodedObservation.view(1, -1)), -1)
            )

            # Actor suggestion
            fullState = torch.cat((recurrentState, latentState), -1)
            action = dreamer.actor(fullState)
            action_np = action.detach().cpu().numpy().reshape(-1)

            # Inject Gaussian action noise if requested
            if noise_sigma > 0.0:
                action_np = action_np + rng.normal(
                    0.0, noise_sigma, size=action_np.shape
                )

            # Track action norm (after noise)
            action_norms.append(float(np.linalg.norm(action_np, ord=2)))

            # Step environment
            next_obs, reward, done = env.step(action_np)

            # Update encoded observation
            obs_tensor = (
                torch.from_numpy(np.asarray(next_obs)).float().unsqueeze(0).to(device)
            )
            encodedObservation = dreamer.encoder(
                obs_tensor.view(-1, *dreamer.observationShape)
            )

            currentScore += float(reward)
            stepCount += 1
            obs = next_obs

        # For CarRacing we don't have a clear "catastrophic failure" definition.
        # For now we mark all episodes as non-failures and focus on return.
        failed = 0

        if len(action_norms) > 0:
            action_norm_mean = float(np.mean(action_norms))
            action_norm_std = float(np.std(action_norms))
        else:
            action_norm_mean = 0.0
            action_norm_std = 0.0

        episode_result = {
            "timestamp": time.time(),
            "env_name": env_name,
            "checkpoint_path": checkpoint_path,
            "checkpoint_step": checkpoint_step,
            "agent_type": agent_type,
            "seed": seed,
            "noise_sigma": noise_sigma,
            "episode_idx": ep,
            "return": currentScore,
            "length": stepCount,
            "failed": int(failed),
            "action_norm_mean": action_norm_mean,
            "action_norm_std": action_norm_std,
        }
        results.append(episode_result)

        print(
            f"[CarRacing | Ep {ep+1}/{num_episodes}] "
            f"Return={currentScore:.2f}, Len={stepCount}, "
            f"Failed={failed}, ActNormMean={action_norm_mean:.3f}"
        )

    # Append to / create CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    if os.path.exists(output_csv):
        existing = pd.read_csv(output_csv)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(output_csv, index=False)
    print(f"[Eval] Saved {len(results)} CarRacing episodes to {output_csv}")


def main():
    args = parse_args()
    run_eval_carracing(
        config_file=args.config,
        checkpoint_path=args.checkpoint_path,
        checkpoint_step=args.checkpoint_step,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
        output_csv=args.output_csv,
        agent_type=args.agent_type,
    )


if __name__ == "__main__":
    main()
