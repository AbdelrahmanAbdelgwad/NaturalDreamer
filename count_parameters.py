# Take a checkpoint path as input and count the number of parameters in the model
import torch
import argparse
from dreamer import Dreamer
from utils import loadConfig
from envs import (
    DMControlWrapper,
    getEnvProperties,
    CleanGymWrapper,
    GymPixelsProcessingWrapper,
)
import gymnasium as gym
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


import torch


def inspect_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")

    print("=== Checkpoint Keys ===")
    for key in ckpt.keys():
        if isinstance(ckpt[key], dict):
            print(f"\n{key}:")
            for k, v in ckpt[key].items():
                if hasattr(v, "shape"):
                    print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"{key}: {ckpt[key]}")

    # Infer key architecture params
    print("\n=== Inferred Config ===")

    # Recurrent size from GRU hidden state
    recurrent_size = ckpt["recurrentModel"]["recurrent.weight_hh"].shape[0] // 3
    print(f"recurrentSize: {recurrent_size}")

    # Latent size from priorNet output
    prior_out = ckpt["priorNet"]["network.4.weight"].shape[0]  # last linear layer
    print(f"latentSize (length * classes): {prior_out}")

    # Actor hidden size
    actor_hidden = ckpt["actor"]["network.0.weight"].shape[0]
    print(f"actor hiddenSize: {actor_hidden}")

    # Action size (actor output / 2 for mean+std)
    action_size = ckpt["actor"]["network.4.weight"].shape[0] // 2
    print(f"actionSize: {action_size}")

    # Check if image or vector encoder
    if "convolutionalNet.0.weight" in ckpt["encoder"]:
        print("Encoder type: Convolutional (image obs)")
    else:
        print("Encoder type: Vector")


def count_parameters(dreamer):
    total = 0
    components = [
        dreamer.encoder,
        dreamer.decoder,
        dreamer.recurrentModel,
        dreamer.priorNet,
        dreamer.posteriorNet,
        dreamer.rewardPredictor,
        dreamer.actor,
        dreamer.critic,
    ]
    if dreamer.config.useContinuationPrediction:
        components.append(dreamer.continuePredictor)

    for comp in components:
        total += sum(p.numel() for p in comp.parameters() if p.requires_grad)
    return total / 1e6  # Return in millions


def main(args):
    config = loadConfig(args.config_file)

    if "CarRacing" in config.environmentName:
        env = CleanGymWrapper(
            GymPixelsProcessingWrapper(
                gym.wrappers.ResizeObservation(
                    gym.make(config.environmentName), (64, 64)
                )
            )
        )
    else:  # DMControl environments
        domain, task = config.environmentName.split("-")
        env = CleanGymWrapper(DMControlWrapper(domain, task))

    observationShape, actionSize, actionLow, actionHigh = getEnvProperties(env)

    dreamer = Dreamer(
        observationShape,
        actionSize,
        actionLow,
        actionHigh,
        device,
        config.dreamer,
    )

    total_params = count_parameters(dreamer)
    print(f"Total trainable parameters in the Dreamer model: {total_params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count parameters in Dreamer model")
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    main(args)
    # inspect_checkpoint(
    #     "/home/abdelrahman/Downloads/cartpole-swingup_CartpoleSwingup-1_2k.pth"
    # )
