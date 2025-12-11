# This script loads the dreamer lyapunov agent and plots a 3D lyapunov function over the true state space.
# Of course the state space (obs space) is [x, cosθ, sinθ, x˙, θ˙] so it could be argued that the lyapunov function
# is not really defined over the true state space, but it's close enough for visualization purposes.
# 1- Plot it over variations in the angle θ and angular velocity θ˙, keeping x and x˙ fixed at 0.
# 2- Plot it over variations in the cart position x and velocity x˙, keeping θ and θ˙ fixed at 0.
# 3- Plot it over variations in x and θ, keeping x˙ and θ˙ fixed at 0.
from sklearn.decomposition import PCA
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lyapunov.dreamer_lyapunov import DreamerLyapunov
from utils import loadConfig, seedEverything
from envs import CleanGymWrapper, DMControlWrapper, GymPixelsProcessingWrapper
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def plot_lyapunov_function(
    agent, env, fixed_state, var1_range, var2_range, var1_name, var2_name, filename
):
    V_values = np.zeros((len(var1_range), len(var2_range)))

    for i, var1 in enumerate(var1_range):
        for j, var2 in enumerate(var2_range):
            state = fixed_state.copy()
            if var1_name == "theta":
                state[1] = np.cos(var1)
                state[2] = np.sin(var1)
            elif var1_name == "theta_dot":
                state[4] = var1
            elif var1_name == "x":
                state[0] = var1
            elif var1_name == "x_dot":
                state[3] = var1

            if var2_name == "theta":
                state[1] = np.cos(var2)
                state[2] = np.sin(var2)
            elif var2_name == "theta_dot":
                state[4] = var2
            elif var2_name == "x":
                state[0] = var2
            elif var2_name == "x_dot":
                state[3] = var2

            obs = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                encoded_state = agent.encoder(obs)
                h = torch.zeros(1, agent.recurrentSize, device=agent.device)
                z, _ = agent.posteriorNet(torch.cat((h, encoded_state), -1))
                V = agent.lyapunovModel(z).cpu().numpy()
            V_values[i, j] = V

    X, Y = np.meshgrid(var1_range, var2_range)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, V_values.T, cmap="viridis")
    ax.set_xlabel(var1_name)
    ax.set_ylabel(var2_name)
    ax.set_zlabel("Lyapunov Function V(z)")
    plt.savefig(filename)
    plt.close()


def main(configFile):
    config = loadConfig(configFile)
    seedEverything(config.seed)

    runName = f"{config.environmentName}_{config.runName}"
    checkpointToLoad = os.path.join(
        config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}"
    )
    print(f"Loading checkpoint from: {checkpointToLoad}")

    if "cartpole-swingup" in config.environmentName:
        domain, task = config.environmentName.split("-")
        env = CleanGymWrapper(DMControlWrapper(domain, task))
    else:
        raise NotImplementedError("This script only supports CartPole environments.")

    observationShape = env.observation_space.shape
    actionSize = env.action_space.shape[0]
    actionLow = env.action_space.low
    actionHigh = env.action_space.high

    agent = DreamerLyapunov(
        observationShape,
        actionSize,
        actionLow,
        actionHigh,
        device,
        config.dreamer,
    )

    agent.loadCheckpoint(checkpointToLoad)

    # Fixed state: [x, cosθ, sinθ, x˙, θ˙] = [0, 1, 0, 0, 0] (upright position)
    fixed_state = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

    # 1- Vary θ and θ˙
    theta_range = np.linspace(-np.pi, np.pi, 50)
    theta_dot_range = np.linspace(-5.0, 5.0, 50)
    plot_lyapunov_function(
        agent,
        env,
        fixed_state,
        theta_range,
        theta_dot_range,
        "theta",
        "theta_dot",
        "lyapunov_theta_theta_dot.png",
    )

    # 2- Vary x and x˙
    x_range = np.linspace(-2.0, 2.0, 50)
    x_dot_range = np.linspace(-5.0, 5.0, 50)
    plot_lyapunov_function(
        agent,
        env,
        fixed_state,
        x_range,
        x_dot_range,
        "x",
        "x_dot",
        "lyapunov_x_x_dot.png",
    )

    # 3- Vary x and θ
    plot_lyapunov_function(
        agent,
        env,
        fixed_state,
        x_range,
        theta_range,
        "x",
        "theta",
        "lyapunov_x_theta.png",
    )

    # Use PCA to plot over first two principal components of the state space
    
    # Generate diverse states for PCA fitting
    print("Generating states for PCA...")
    n_samples = 1000
    states_for_pca = []
    for _ in range(n_samples):
        state = np.array([
            np.random.uniform(-2.0, 2.0),      # x
            np.cos(np.random.uniform(-np.pi, np.pi)),  # cos(theta)
            np.sin(np.random.uniform(-np.pi, np.pi)),  # sin(theta)
            np.random.uniform(-5.0, 5.0),      # x_dot
            np.random.uniform(-5.0, 5.0),      # theta_dot
        ])
        states_for_pca.append(state)
    states_for_pca = np.array(states_for_pca)
    
    # Fit PCA
    pca = PCA(n_components=2)
    pca.fit(states_for_pca)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    
    # Create grid in PC space
    pc1_range = np.linspace(-3, 3, 50)
    pc2_range = np.linspace(-3, 3, 50)
    PC1, PC2 = np.meshgrid(pc1_range, pc2_range)
    
    V_values_pca = np.zeros_like(PC1)
    
    print("Computing Lyapunov values over PC space...")
    for i in range(len(pc1_range)):
        for j in range(len(pc2_range)):
            # Transform from PC space back to state space
            pc_coords = np.array([PC1[j, i], PC2[j, i]])
            state = pca.inverse_transform(pc_coords)
            
            # Normalize theta components to unit circle
            theta_magnitude = np.sqrt(state[1]**2 + state[2]**2)
            if theta_magnitude > 1e-6:
                state[1] /= theta_magnitude
                state[2] /= theta_magnitude
            
            obs = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                encoded_state = agent.encoder(obs)
                h = torch.zeros(1, agent.recurrentSize, device=agent.device)
                z, _ = agent.posteriorNet(torch.cat((h, encoded_state), -1))
                V = agent.lyapunovModel(z).cpu().numpy()
            V_values_pca[j, i] = V
    
    # Plot as 3D surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(PC1, PC2, V_values_pca, cmap="viridis")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Lyapunov Function V(z)")
    ax.set_title("Lyapunov Function over PCA State Space")
    plt.savefig("lyapunov_pca.png")
    plt.close()
    
    print("PCA plot saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cartpole_lyapunov.yml",
        help="Path to the config file.",
    )
    args = parser.parse_args()
    main(args.config)
