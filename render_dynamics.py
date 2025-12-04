import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

import gymnasium as gym

from lyapunov.dreamer_lyapunov import DreamerLyapunov
from utils import loadConfig, seedEverything, ensureParentFolders
from envs import (
    DMControlWrapper,
    getEnvProperties,
    GymPixelsProcessingWrapper,
    CleanGymWrapper,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------
# Controllers: learned actor, LQR, and energy-based swing-up
# ------------------------------------------------------------


def actor_controller(dreamer, actor_state, obs):
    """
    Controller that uses the learned actor.
    actor_state is a dict holding recurrent/latent/action/encodedObs
    for closed-loop actor updates based on env observations.
    """
    with torch.no_grad():
        # Encode observation
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dreamer.device)
        encodedObservation = dreamer.encoder(obs_t)

        # Unpack state
        recurrentState = actor_state["recurrentState"]
        latentState = actor_state["latentState"]
        prev_action = actor_state["prev_action"]

        # Update recurrent + posterior as in environmentInteraction()
        recurrentState = dreamer.recurrentModel(
            recurrentState, latentState, prev_action
        )
        latentState, _ = dreamer.posteriorNet(
            torch.cat((recurrentState, encodedObservation.view(1, -1)), -1)
        )
        fullState = torch.cat((recurrentState, latentState), -1)

        action = dreamer.actor(fullState)
        action_np = action.cpu().numpy().reshape(-1)

        # Update actor internal state
        actor_state["recurrentState"] = recurrentState
        actor_state["latentState"] = latentState
        actor_state["prev_action"] = action

    return action_np


def cartpole_extract_state(obs):
    """
    Try to extract [x, x_dot, theta, theta_dot] from observation.
    - If 4D, assume Gym CartPole: [x, x_dot, theta, theta_dot]
    - If DMControl style (cos, sin, ...), approximate theta.
    You may need to adapt this to your exact env.
    """
    obs = np.asarray(obs)
    if obs.shape[0] == 4:
        x, x_dot, theta, theta_dot = obs
    else:
        # Heuristic for DMControl cartpole: [cos(theta), sin(theta), x, x_dot, theta_dot]
        cos_th, sin_th = obs[0], obs[1]
        theta = np.arctan2(sin_th, cos_th)
        # Guess remaining ordering
        x = obs[2]
        x_dot = obs[3]
        theta_dot = obs[4] if obs.shape[0] > 4 else 0.0
    return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)


def lqr_controller(obs, max_force=1.0):
    """
    Simple LQR-like linear feedback around upright for cartpole.
    This is a heuristic K; tune for your system.

    u = -K * [x, x_dot, theta, theta_dot]
    """
    state = cartpole_extract_state(obs)

    # Example gains (roughly stabilizing for classic CartPole)
    K = np.array([1.0, 1.0, 10.0, 1.0], dtype=np.float32)
    u = -K.dot(state)

    u = np.clip(u, -max_force, max_force)
    return np.array([u], dtype=np.float32)


def energy_swingup_controller(obs, max_force=1.0):
    """
    Energy-based swing-up controller for cartpole (heuristic).
    You should adapt m, l, etc. to your actual env parameters.
    """
    x, x_dot, theta, theta_dot = cartpole_extract_state(obs)

    # Parameters (approx CartPole)
    m = 0.1  # pole mass
    l = 0.5  # half pole length
    g = 9.81

    # Mechanical energy of the pole (relative to upright)
    E = 0.5 * m * (l**2) * (theta_dot**2) + m * g * l * (1 - np.cos(theta))
    E_target = 2 * m * g * l  # energy at upright (approx)

    k = 1.0  # energy gain
    u = k * (E - E_target) * np.sign(theta_dot * np.cos(theta))

    # Damping on cart velocity to avoid runaway
    u -= 0.1 * x_dot

    u = np.clip(u, -max_force, max_force)
    return np.array([u], dtype=np.float32)


# ------------------------------------------------------------
# Trajectory collection from real env
# ------------------------------------------------------------


def collect_real_trajectory(env, dreamer, controller_type, horizon, seed=None):
    """
    Run the real environment with chosen controller and record
    observations & actions.
    """
    if seed is not None:
        obs = env.reset(seed=seed)
    else:
        obs = env.reset()
    # CleanGymWrapper returns obs directly (as in main_lyapunov.py)
    # If your wrapper returns (obs, info), adapt accordingly.

    obs_list = [np.array(obs)]
    act_list = []

    # State for actor controller
    actor_state = {
        "recurrentState": torch.zeros(1, dreamer.recurrentSize, device=dreamer.device),
        "latentState": torch.zeros(1, dreamer.latentSize, device=dreamer.device),
        "prev_action": torch.zeros(1, dreamer.actionSize, device=dreamer.device),
    }

    for t in range(horizon):
        if controller_type == "actor":
            action = actor_controller(dreamer, actor_state, obs)
        elif controller_type == "lqr":
            action = lqr_controller(obs)
        elif controller_type == "swingup":
            action = energy_swingup_controller(obs)
        else:
            raise ValueError(f"Unknown controller_type: {controller_type}")

        next_obs, reward, done = env.step(action)
        obs_list.append(np.array(next_obs))
        act_list.append(action)

        obs = next_obs
        if done:
            break

    return np.stack(obs_list, axis=0), np.stack(act_list, axis=0)


# ------------------------------------------------------------
# World model rollout (prediction) given same actions
# ------------------------------------------------------------


@torch.no_grad()
def rollout_world_model(dreamer, obs0, actions):
    """
    Rollout the learned world model starting from obs0 and driven
    by the provided action sequence.

    Uses encoder + posterior at t=0, then recurrentModel + priorNet +
    decoder for forward prediction, as in imagination. :contentReference[oaicite:1]{index=1}
    """
    obs_shape = dreamer.observationShape
    is_image = len(obs_shape) == 3

    # Shortcuts
    encoder = dreamer.encoder
    decoder = dreamer.decoder
    recurrentModel = dreamer.recurrentModel
    priorNet = dreamer.priorNet
    posteriorNet = dreamer.posteriorNet

    # Initial recurrent/latent state
    recurrentState = torch.zeros(1, dreamer.recurrentSize, device=dreamer.device)
    latentState = torch.zeros(1, dreamer.latentSize, device=dreamer.device)

    obs0_t = torch.from_numpy(obs0).float().unsqueeze(0).to(dreamer.device)
    encoded0 = encoder(obs0_t)
    latentState, _ = posteriorNet(torch.cat((recurrentState, encoded0), -1))
    fullState = torch.cat((recurrentState, latentState), -1)

    # Decode initial observation
    decoded0 = decoder(fullState).view(1, *obs_shape).cpu().numpy()[0]
    model_obs = [decoded0]

    # Rollout using prior dynamics
    for a in actions:
        a_t = torch.from_numpy(a).float().unsqueeze(0).to(dreamer.device)
        recurrentState = recurrentModel(recurrentState, latentState, a_t)
        latentState, _ = priorNet(recurrentState)
        fullState = torch.cat((recurrentState, latentState), -1)
        decoded = decoder(fullState).view(1, *obs_shape).cpu().numpy()[0]
        model_obs.append(decoded)

    return np.stack(model_obs, axis=0)  # (T+1, *obs_shape)


# ------------------------------------------------------------
# Visualization utilities
# ------------------------------------------------------------


def plot_vector_trajectory(real_obs, model_obs, save_path=None, title=""):
    """
    For low-dimensional vector observations: plot each state component
    real vs model over time.
    """
    T, dim = real_obs.shape
    t = np.arange(T)

    fig, axes = plt.subplots(dim, 1, figsize=(8, 2 * dim), sharex=True)
    if dim == 1:
        axes = [axes]

    for i in range(dim):
        axes[i].plot(t, real_obs[:, i], label="real")
        axes[i].plot(t, model_obs[:, i], "--", label="model")
        axes[i].set_ylabel(f"state[{i}]")
        axes[i].grid(True)
    axes[-1].set_xlabel("time step")
    axes[0].legend()
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    plt.close(fig)


def make_side_by_side_video(real_obs_imgs, model_obs_imgs, filename, fps=30):
    """
    For pixel observations: create a video with real vs model frames
    side-by-side.
    real_obs_imgs, model_obs_imgs: (T, H, W, C) arrays in [0,1] or [0,255]
    """
    T = min(real_obs_imgs.shape[0], model_obs_imgs.shape[0])
    frames = []
    for t in range(T):
        real = real_obs_imgs[t]
        model = model_obs_imgs[t]

        if real.dtype != np.uint8:
            real = np.clip(real * 255.0, 0, 255).astype(np.uint8)
        if model.dtype != np.uint8:
            model = np.clip(model * 255.0, 0, 255).astype(np.uint8)

        # Resize / pad if needed
        h = max(real.shape[0], model.shape[0])
        w = max(real.shape[1], model.shape[1])

        def pad(img):
            pad_h = h - img.shape[0]
            pad_w = w - img.shape[1]
            return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        real = pad(real)
        model = pad(model)

        concat = np.concatenate([real, model], axis=1)
        frames.append(concat)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with imageio.get_writer(filename, fps=fps) as writer:
        for f in frames:
            writer.append_data(f)


# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------


def make_env_from_config(config):
    if "CarRacing" in config.environmentName:
        env = CleanGymWrapper(
            GymPixelsProcessingWrapper(
                gym.wrappers.ResizeObservation(
                    gym.make(config.environmentName), (64, 64)
                )
            )
        )
    else:
        domain, task = config.environmentName.split("-")
        env = CleanGymWrapper(DMControlWrapper(domain, task))
    return env


def main(configFile, controller_type, horizon, checkpoint_suffix):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName = f"{config.environmentName}_{config.runName}"
    checkpointBase = os.path.join(config.folderNames.checkpointsFolder, runName)
    if checkpoint_suffix is None:
        checkpointPath = f"{checkpointBase}_{config.checkpointToLoad}"
    else:
        checkpointPath = f"{checkpointBase}_{checkpoint_suffix}"
    if not checkpointPath.endswith(".pth"):
        checkpointPath += ".pth"

    out_plot_base = os.path.join(
        config.folderNames.plotsFolder, runName + "_model_eval"
    )
    out_video_base = os.path.join(
        config.folderNames.videosFolder, runName + "_model_eval"
    )
    ensureParentFolders(out_plot_base, out_video_base, checkpointBase, out_video_base)

    env = make_env_from_config(config)
    observationShape, actionSize, actionLow, actionHigh = getEnvProperties(env)
    print(
        f"envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}"
    )

    dreamer = DreamerLyapunov(
        observationShape, actionSize, actionLow, actionHigh, device, config.dreamer
    )
    dreamer.loadCheckpoint(checkpointPath)
    print(f"Loaded checkpoint from {checkpointPath}")

    # Collect one real trajectory
    real_obs, actions = collect_real_trajectory(
        env, dreamer, controller_type, horizon, seed=config.seed
    )

    # Rollout world model with the same actions (starting from real_obs[0])
    model_obs = rollout_world_model(dreamer, real_obs[0], actions)

    # Align lengths
    T = min(real_obs.shape[0], model_obs.shape[0])
    real_obs = real_obs[:T]
    model_obs = model_obs[:T]

    is_image = len(observationShape) == 3

    if not is_image:
        # Vector observations: plot components and compute error
        mse = np.mean((real_obs - model_obs) ** 2, axis=0)
        print("Per-dimension MSE between real and model:", mse)

        plot_path = f"{out_plot_base}_{controller_type}.png"
        plot_vector_trajectory(
            real_obs,
            model_obs,
            save_path=plot_path,
            title=f"Dynamics model vs env ({controller_type})",
        )
        print(f"Saved state comparison plot to {plot_path}")
    else:
        # Pixel observations: build side-by-side video
        video_path = f"{out_video_base}_{controller_type}.mp4"
        make_side_by_side_video(real_obs, model_obs, filename=video_path, fps=30)
        print(f"Saved side-by-side model vs env video to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cartpole_lyapunov.yml")
    parser.add_argument(
        "--controller",
        type=str,
        default="actor",
        choices=["actor", "lqr", "swingup"],
        help="Which controller to use to drive env and model.",
    )
    parser.add_argument("--horizon", type=int, default=500, help="Max rollout horizon.")
    parser.add_argument(
        "--checkpoint_suffix",
        type=str,
        default=None,
        help="Suffix for checkpoint (e.g., '100k'). If None, uses config.checkpointToLoad.",
    )
    args = parser.parse_args()

    main(
        configFile=args.config,
        controller_type=args.controller,
        horizon=args.horizon,
        checkpoint_suffix=args.checkpoint_suffix,
    )
