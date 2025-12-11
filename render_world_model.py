"""
Compare actual environment rollout vs world model imagination.
Both start from same state, use same actions, but world model uses its own predictions.
"""

import torch
import numpy as np
import argparse
import imageio
import cv2
from dreamer import Dreamer
from utils import loadConfig, seedEverything
from envs import (
    DMControlWrapper,
    CleanGymWrapper,
    getEnvProperties,
    GymPixelsProcessingWrapper,
)
import gymnasium as gym


def compare_rollout(
    dreamer,
    env,
    warmup_steps=50,
    rollout_steps=200,
    seed=42,
    output_path="comparison.mp4",
    fps=30,
    anchor_freq=10,
):
    """
    Run parallel rollout of actual env and world model imagination.

    Args:
        dreamer: Trained Dreamer model
        env: Environment (wrapped)
        warmup_steps: Steps to run before starting comparison (avoid weird initial states)
        rollout_steps: Steps to compare
        seed: Random seed
        output_path: Where to save video
        fps: Video frame rate
        anchor_freq: Frequency of anchoring imagined states to actual environment states

    """
    device = dreamer.device

    # Initialize states
    recurrent_state = torch.zeros(1, dreamer.recurrentSize, device=device)
    latent_state = torch.zeros(1, dreamer.latentSize, device=device)
    action = torch.zeros(1, dreamer.actionSize, device=device)

    # Reset env
    obs = env.reset(seed=seed)

    # --- Warmup Phase ---
    print(f"Running {warmup_steps} warmup steps...")
    for _ in range(warmup_steps):
        encoded_obs = dreamer.encoder(
            torch.from_numpy(obs).float().unsqueeze(0).to(device)
        )
        recurrent_state = dreamer.recurrentModel(recurrent_state, latent_state, action)
        latent_state, _ = dreamer.posteriorNet(
            torch.cat((recurrent_state, encoded_obs.view(1, -1)), -1)
        )
        action = dreamer.actor(torch.cat((recurrent_state, latent_state), -1))
        obs, _, done = env.step(action.cpu().numpy().reshape(-1))
        if done:
            obs = env.reset(seed=seed)
            recurrent_state = torch.zeros(1, dreamer.recurrentSize, device=device)
            latent_state = torch.zeros(1, dreamer.latentSize, device=device)

    print("Warmup complete. Starting comparison rollout...")

    # --- Sync starting point ---
    # Get encoded observation for starting state
    encoded_obs = dreamer.encoder(torch.from_numpy(obs).float().unsqueeze(0).to(device))

    # Update recurrent state with last action
    recurrent_state = dreamer.recurrentModel(recurrent_state, latent_state, action)
    latent_state, _ = dreamer.posteriorNet(
        torch.cat((recurrent_state, encoded_obs.view(1, -1)), -1)
    )

    # Initialize imagined trajectory (starts same as actual)
    imagined_recurrent = recurrent_state.clone()
    imagined_latent = latent_state.clone()

    frames = []

    # --- Comparison Rollout ---
    for step in range(rollout_steps):
        full_state = torch.cat((recurrent_state, latent_state), -1)

        # Get action from actor (same action for both actual and imagined)
        action = dreamer.actor(full_state)
        action_np = action.cpu().numpy().reshape(-1)

        # === Actual Environment Step ===
        obs, reward, done = env.step(action_np)
        actual_frame = env.render()

        if done:
            print(f"Episode ended at step {step}")
            break

        # Update actual trajectory states
        encoded_obs = dreamer.encoder(
            torch.from_numpy(obs).float().unsqueeze(0).to(device)
        )
        recurrent_state = dreamer.recurrentModel(recurrent_state, latent_state, action)
        latent_state, _ = dreamer.posteriorNet(
            torch.cat((recurrent_state, encoded_obs.view(1, -1)), -1)
        )

        # === Imagined World Model Step ===
        # Use priorNet (no observation) instead of posteriorNet
        imagined_recurrent = dreamer.recurrentModel(
            imagined_recurrent, imagined_latent, action
        )
        imagined_latent, _ = dreamer.priorNet(imagined_recurrent)

        # Decode imagined observation
        imagined_full_state = torch.cat((imagined_recurrent, imagined_latent), -1)
        imagined_obs = dreamer.decoder(imagined_full_state)

        # Convert imagined observation to frame
        imagined_frame = obs_to_frame(
            imagined_obs, dreamer.isImageObs, dreamer.observationShape
        )

        # Resize frames to match
        actual_frame, imagined_frame = resize_frames_to_match(
            actual_frame, imagined_frame
        )

        # Create side-by-side frame with labels
        combined = create_side_by_side(actual_frame, imagined_frame, step)
        frames.append(combined)

        if step % 50 == 0:
            print(f"Step {step}/{rollout_steps}")

        # Anchor imagined states to actual every anchor_freq steps
        if (step + 1) % anchor_freq == 0:
            imagined_recurrent = recurrent_state.clone()
            imagined_latent = latent_state.clone()

    # Save video
    print(f"Saving video to {output_path}")
    save_video(frames, output_path, fps)
    print("Done!")
    return output_path


def obs_to_frame(decoded_obs, is_image_obs, obs_shape):
    """Convert decoded observation tensor to RGB frame."""
    obs_np = decoded_obs.squeeze(0).cpu().numpy()

    if is_image_obs:
        # Shape: (C, H, W) -> (H, W, C)
        obs_np = np.transpose(obs_np, (1, 2, 0))
        # Clip and scale to 0-255
        obs_np = np.clip(obs_np * 255, 0, 255).astype(np.uint8)
    else:
        # Vector observation - create a simple visualization
        # Normalize and create a bar chart style visualization
        obs_np = obs_np.flatten()
        h, w = 240, 320
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Normalize values to [0, 1]
        obs_min, obs_max = obs_np.min(), obs_np.max()
        if obs_max - obs_min > 1e-6:
            obs_norm = (obs_np - obs_min) / (obs_max - obs_min)
        else:
            obs_norm = np.zeros_like(obs_np) + 0.5

        # Draw bars
        n_bars = len(obs_norm)
        bar_width = max(1, w // (n_bars + 1))
        for i, val in enumerate(obs_norm):
            x = int(i * bar_width + bar_width // 2)
            bar_height = int(val * (h - 40))
            cv2.rectangle(
                frame,
                (x, h - 20 - bar_height),
                (x + bar_width - 2, h - 20),
                (0, 255, int(255 * val)),
                -1,
            )
        obs_np = frame

    return obs_np


def resize_frames_to_match(frame1, frame2, target_height=240):
    """Resize frames to same height while preserving aspect ratio."""
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    # Resize both to target height
    scale1 = target_height / h1
    scale2 = target_height / h2

    new_w1 = int(w1 * scale1)
    new_w2 = int(w2 * scale2)

    frame1_resized = cv2.resize(frame1, (new_w1, target_height))
    frame2_resized = cv2.resize(frame2, (new_w2, target_height))

    return frame1_resized, frame2_resized


def create_side_by_side(actual_frame, imagined_frame, step):
    """Create side-by-side comparison frame with labels."""
    h1, w1 = actual_frame.shape[:2]
    h2, w2 = imagined_frame.shape[:2]

    # Add padding for labels
    label_height = 30
    gap = 10

    total_width = w1 + gap + w2
    total_height = max(h1, h2) + label_height

    combined = np.zeros((total_height, total_width, 3), dtype=np.uint8)

    # Place frames
    combined[label_height : label_height + h1, :w1] = actual_frame
    combined[label_height : label_height + h2, w1 + gap :] = imagined_frame

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        combined, f"Actual (step {step})", (10, 22), font, 0.6, (255, 255, 255), 1
    )
    cv2.putText(
        combined, f"Imagined", (w1 + gap + 10, 22), font, 0.6, (255, 255, 255), 1
    )

    # Add vertical separator
    cv2.line(
        combined, (w1 + gap // 2, 0), (w1 + gap // 2, total_height), (128, 128, 128), 1
    )

    return combined


def save_video(frames, output_path, fps):
    """Save frames as video with proper dimensions."""
    if not frames:
        print("No frames to save!")
        return

    # Ensure dimensions are divisible by macro_block_size (16)
    h, w = frames[0].shape[:2]
    target_h = ((h + 15) // 16) * 16
    target_w = ((w + 15) // 16) * 16

    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            if frame.shape[0] != target_h or frame.shape[1] != target_w:
                frame = np.pad(
                    frame,
                    (
                        (0, target_h - frame.shape[0]),
                        (0, target_w - frame.shape[1]),
                        (0, 0),
                    ),
                    mode="edge",
                )
            writer.append_data(frame)


def main():
    parser = argparse.ArgumentParser(description="Compare actual vs imagined rollout")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup steps")
    parser.add_argument("--steps", type=int, default=200, help="Rollout steps")
    parser.add_argument(
        "--output", type=str, default="comparison.mp4", help="Output video path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument(
        "--anchor-freq",
        type=int,
        default=10,
        help="Anchor frequency to to decide how long the dream interval should be before"
        " resetting to real environment state and continuing dreaming from there.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config = loadConfig(args.config)
    seedEverything(args.seed)

    # Create environment
    if "CarRacing" in config.environmentName:
        env = CleanGymWrapper(
            GymPixelsProcessingWrapper(
                gym.wrappers.ResizeObservation(
                    gym.make(config.environmentName, render_mode="rgb_array"), (64, 64)
                )
            )
        )
    else:  # DMControl environments
        domain, task = config.environmentName.split("-")
        env = CleanGymWrapper(DMControlWrapper(domain, task))

    obs_shape, action_size, action_low, action_high = getEnvProperties(env)

    print(f"Environment: {config.environmentName}")
    print(f"Observation shape: {obs_shape}, Action size: {action_size}")

    # Create and load Dreamer
    dreamer = Dreamer(
        obs_shape, action_size, action_low, action_high, device, config.dreamer
    )
    dreamer.loadCheckpoint(args.checkpoint)

    # Set to eval mode
    dreamer.encoder.eval()
    dreamer.decoder.eval()
    dreamer.recurrentModel.eval()
    dreamer.priorNet.eval()
    dreamer.posteriorNet.eval()
    dreamer.actor.eval()

    # Run comparison
    with torch.no_grad():
        compare_rollout(
            dreamer,
            env,
            warmup_steps=args.warmup,
            rollout_steps=args.steps,
            seed=args.seed,
            output_path=args.output,
            fps=args.fps,
            anchor_freq=args.anchor_freq,
        )


if __name__ == "__main__":
    main()
