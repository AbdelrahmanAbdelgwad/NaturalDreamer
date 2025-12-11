# TODO: Add wandb integration for experiment tracking
# TODO: Add print statements for key steps and metrics
# TODO:
import gymnasium as gym
import torch
import argparse
import os
from dreamer import Dreamer
from utils import loadConfig, seedEverything, plotMetrics
from envs import (
    DMControlWrapper,
    DMControlWrapper,
    getEnvProperties,
    GymPixelsProcessingWrapper,
    CleanGymWrapper,
)
from utils import saveLossesToCSV, ensureParentFolders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Remove warning messages from all libraries
import warnings

warnings.filterwarnings("ignore")

# import weights & biases
import wandb


def main(configFile):

    wandb.init(project="NaturalDreamer", name=configFile)

    print(f"Loading config from {configFile}")
    config = loadConfig(configFile)

    wandb.config.update(config)

    print(f"Setting random seed to {config.seed}")
    seedEverything(config.seed)

    runName = f"{config.environmentName}_{config.runName}"
    checkpointToLoad = os.path.join(
        config.folderNames.checkpointsFolder, f"{runName}_{config.checkpointToLoad}"
    )
    metricsFilename = os.path.join(config.folderNames.metricsFolder, runName)
    plotFilename = os.path.join(config.folderNames.plotsFolder, runName)
    checkpointFilenameBase = os.path.join(config.folderNames.checkpointsFolder, runName)
    videoFilenameBase = os.path.join(config.folderNames.videosFolder, runName)
    ensureParentFolders(
        metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase
    )

    if "CarRacing" in config.environmentName:
        env = CleanGymWrapper(
            GymPixelsProcessingWrapper(
                gym.wrappers.ResizeObservation(
                    gym.make(config.environmentName), (64, 64)
                )
            )
        )
        envEvaluation = CleanGymWrapper(
            GymPixelsProcessingWrapper(
                gym.wrappers.ResizeObservation(
                    gym.make(config.environmentName, render_mode="rgb_array"), (64, 64)
                )
            )
        )
    else:  # DMControl environments
        domain, task = config.environmentName.split("-")
        env = CleanGymWrapper(DMControlWrapper(domain, task))
        envEvaluation = CleanGymWrapper(DMControlWrapper(domain, task))

    observationShape, actionSize, actionLow, actionHigh = getEnvProperties(env)
    print("Starting Dreamer training...")
    print(f"Environment: {config.environmentName}")
    print(
        f"envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}"
    )

    dreamer = Dreamer(
        observationShape, actionSize, actionLow, actionHigh, device, config.dreamer
    )
    if config.resume:
        print(f"Resuming training from checkpoint: {checkpointToLoad}")
        dreamer.loadCheckpoint(checkpointToLoad)
    else:
        print("Starting training from scratch.")

    # Initial environment interaction to fill the replay buffer before training the world model
    # Remember that we train the world model first, then the behavior policy using the learned world model
    print("Starting initial environment interaction to fill the replay buffer...")
    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)
    print("Initial environment interaction completed.")

    iterationsNum = config.gradientSteps // config.replayRatio
    print("Starting main training loop...")
    print(
        f"Total gradient steps to perform: {config.gradientSteps} = {iterationsNum} iterations x {config.replayRatio} replay ratio"
    )
    print("Beginning training iterations...")

    print(
        f"Will be saving checkpoints every {config.checkpointInterval} gradient steps."
    )

    for iteration in range(iterationsNum):
        print(
            f"Iteration {iteration + 1} of {iterationsNum}, Total Gradient Steps: {dreamer.totalGradientSteps}"
        )
        for replayStep in range(config.replayRatio):

            sampledData = dreamer.buffer.sample(
                dreamer.config.batchSize, dreamer.config.batchLength
            )
            initialStates, worldModelMetrics = dreamer.worldModelTraining(sampledData)
            behaviorMetrics = dreamer.behaviorTraining(initialStates)
            dreamer.totalGradientSteps += 1

            if (
                dreamer.totalGradientSteps % config.checkpointInterval == 0
                and config.saveCheckpoints
            ):
                suffix = f"{dreamer.totalGradientSteps/1000:.2f}k"
                dreamer.saveCheckpoint(f"{checkpointFilenameBase}_{suffix}")
                evaluationScore = dreamer.environmentInteraction(
                    envEvaluation,
                    config.numEvaluationEpisodes,
                    seed=config.seed,
                    evaluation=True,
                    saveVideo=True,
                    filename=f"{videoFilenameBase}_{suffix}",
                )
                print(
                    f"Saved Checkpoint and Video at {suffix:>6} gradient steps. Evaluation score: {evaluationScore:>8.2f}"
                )

        mostRecentScore = dreamer.environmentInteraction(
            env, config.numInteractionEpisodes, seed=config.seed
        )
        if config.saveMetrics:
            metricsBase = {
                "envSteps": dreamer.totalEnvSteps,
                "gradientSteps": dreamer.totalGradientSteps,
                "totalReward": mostRecentScore,
            }
            saveLossesToCSV(
                metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics
            )
            plotMetrics(
                f"{metricsFilename}",
                savePath=f"{plotFilename}",
                title=f"{config.environmentName}",
            )

            # log to wandb as reward, world model loss, behavior loss
            # logging should make them appear as curves over envSteps
            wandb.log(
                {
                    "envSteps": dreamer.totalEnvSteps,
                    "gradientSteps": dreamer.totalGradientSteps,
                    "totalReward": mostRecentScore,
                    **worldModelMetrics,
                    **behaviorMetrics,
                }
            )

    print("Training completed.")


if __name__ == "__main__":
    from time import time

    t1 = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="car-racing-v3.yml")
    main(parser.parse_args().config)
    t2 = time()
    print(f"Total execution time: {t2 - t1:.2f} seconds")
