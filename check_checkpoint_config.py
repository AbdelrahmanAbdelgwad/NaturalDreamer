"""
Compare a checkpoint against a config file to check compatibility.
Reports shape mismatches and missing/extra keys.
"""

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


def get_model_state_dicts(dreamer):
    """Extract state dicts from Dreamer components."""
    components = {
        "encoder": dreamer.encoder,
        "decoder": dreamer.decoder,
        "recurrentModel": dreamer.recurrentModel,
        "priorNet": dreamer.priorNet,
        "posteriorNet": dreamer.posteriorNet,
        "rewardPredictor": dreamer.rewardPredictor,
        "actor": dreamer.actor,
        "critic": dreamer.critic,
    }
    if dreamer.config.useContinuationPrediction:
        components["continuePredictor"] = dreamer.continuePredictor

    return {name: comp.state_dict() for name, comp in components.items()}


def compare_state_dicts(ckpt_dict, model_dict, component_name):
    """Compare checkpoint state dict with model state dict."""
    issues = []

    ckpt_keys = set(ckpt_dict.keys())
    model_keys = set(model_dict.keys())

    # Missing in checkpoint
    missing = model_keys - ckpt_keys
    if missing:
        issues.append(f"  Missing in checkpoint: {missing}")

    # Extra in checkpoint
    extra = ckpt_keys - model_keys
    if extra:
        issues.append(f"  Extra in checkpoint: {extra}")

    # Shape mismatches
    common_keys = ckpt_keys & model_keys
    for key in sorted(common_keys):
        ckpt_shape = tuple(ckpt_dict[key].shape)
        model_shape = tuple(model_dict[key].shape)
        if ckpt_shape != model_shape:
            issues.append(f"  {key}: checkpoint {ckpt_shape} vs config {model_shape}")

    return issues


def infer_config_from_checkpoint(ckpt):
    """Infer architecture parameters from checkpoint weights."""
    inferred = {}

    # Recurrent size from GRU (weight_hh shape is 3*hidden x hidden)
    if "recurrentModel" in ckpt:
        hh = ckpt["recurrentModel"].get("recurrent.weight_hh")
        if hh is not None:
            inferred["recurrentSize"] = hh.shape[0] // 3

    # Latent size from priorNet output layer
    if "priorNet" in ckpt:
        for key in sorted(ckpt["priorNet"].keys(), reverse=True):
            if "weight" in key and "network" in key:
                inferred["latentSize"] = ckpt["priorNet"][key].shape[0]
                break

    # Action size from actor output (divided by 2 for mean+std)
    if "actor" in ckpt:
        for key in sorted(ckpt["actor"].keys(), reverse=True):
            if "weight" in key and "network" in key:
                inferred["actionSize"] = ckpt["actor"][key].shape[0] // 2
                break

    # Encoder type
    if "encoder" in ckpt:
        if any("convolutionalNet" in k or "Conv" in k for k in ckpt["encoder"].keys()):
            inferred["encoderType"] = "convolutional"
        else:
            inferred["encoderType"] = "vector"
            # Input size from first layer
            for key in sorted(ckpt["encoder"].keys()):
                if "weight" in key:
                    inferred["obsSize"] = ckpt["encoder"][key].shape[1]
                    break

    # Encoded obs size from encoder output
    if "encoder" in ckpt:
        for key in sorted(ckpt["encoder"].keys(), reverse=True):
            if "weight" in key:
                inferred["encodedObsSize"] = ckpt["encoder"][key].shape[0]
                break

    return inferred


def main(args):
    device = torch.device("cpu")  # CPU is fine for comparison

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Infer config from checkpoint
    print("\n=== Inferred from Checkpoint ===")
    inferred = infer_config_from_checkpoint(ckpt)
    for k, v in inferred.items():
        print(f"  {k}: {v}")

    # Load config and create model
    print(f"\nLoading config: {args.config}")
    config = loadConfig(args.config)

    # Create environment to get shapes
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

    obs_shape, action_size, action_low, action_high = getEnvProperties(env)
    env.env.close()

    print(f"\n=== Config Values ===")
    print(f"  environmentName: {config.environmentName}")
    print(f"  obsShape: {obs_shape}")
    print(f"  actionSize: {action_size}")
    print(f"  recurrentSize: {config.dreamer.recurrentSize}")
    print(f"  latentLength: {config.dreamer.latentLength}")
    print(f"  latentClasses: {config.dreamer.latentClasses}")
    print(f"  latentSize: {config.dreamer.latentLength * config.dreamer.latentClasses}")
    print(f"  encodedObsSize: {config.dreamer.encodedObsSize}")

    # Quick compatibility check before creating model
    if "actionSize" in inferred and inferred["actionSize"] != action_size:
        print(
            f"\n⚠️  Action size mismatch: checkpoint={inferred['actionSize']}, config env={action_size}"
        )

    # Create Dreamer model from config
    print("\nCreating model from config...")
    dreamer = Dreamer(
        obs_shape, action_size, action_low, action_high, device, config.dreamer
    )
    model_dicts = get_model_state_dicts(dreamer)

    # Compare each component
    print("\n=== Comparison Results ===")
    all_good = True

    for component in [
        "encoder",
        "decoder",
        "recurrentModel",
        "priorNet",
        "posteriorNet",
        "rewardPredictor",
        "actor",
        "critic",
        "continuePredictor",
    ]:
        if component not in ckpt:
            if component in model_dicts:
                print(f"\n❌ {component}: Missing from checkpoint")
                all_good = False
            continue

        if component not in model_dicts:
            print(f"\n❌ {component}: In checkpoint but not in config model")
            all_good = False
            continue

        issues = compare_state_dicts(ckpt[component], model_dicts[component], component)

        if issues:
            print(f"\n❌ {component}:")
            for issue in issues:
                print(issue)
            all_good = False
        else:
            print(f"✓ {component}: OK")

    # Summary
    print("\n" + "=" * 40)
    if all_good:
        print("✓ Checkpoint and config are COMPATIBLE")
    else:
        print("❌ Checkpoint and config are INCOMPATIBLE")
        print("\nTo fix, adjust config to match inferred values above.")

    return all_good


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check checkpoint vs config compatibility"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=True,
        help="Path to checkpoint .pth file",
    )
    parser.add_argument(
        "--config", "-f", type=str, required=True, help="Path to config .yml file"
    )
    args = parser.parse_args()
    main(args)
