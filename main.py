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


def main(configFile):
    config = loadConfig(configFile)
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
    print(
        f"envProperties: obs {observationShape}, action size {actionSize}, actionLow {actionLow}, actionHigh {actionHigh}"
    )

    dreamer = Dreamer(
        observationShape, actionSize, actionLow, actionHigh, device, config.dreamer
    )
    if config.resume:
        dreamer.loadCheckpoint(checkpointToLoad)

    # Initial environment interaction to fill the replay buffer before training the world model
    # Remember that we train the world model first, then the behavior policy using the learned world model
    dreamer.environmentInteraction(env, config.episodesBeforeStart, seed=config.seed)

    iterationsNum = config.gradientSteps // config.replayRatio
    for _ in range(iterationsNum):
        for _ in range(config.replayRatio):
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
                suffix = f"{dreamer.totalGradientSteps/1000:.0f}k"
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
            print(
                f"worldModel Loss: {worldModelMetrics['worldModelLoss']:.4f}, "
                f"reconstruction Loss: {worldModelMetrics['reconstructionLoss']:.4f}, "
                f"rewardPredictor Loss: {worldModelMetrics['rewardPredictorLoss']:.4f}, "
                f"KL Loss: {worldModelMetrics['klLoss']:.4f}, "
            )
            saveLossesToCSV(
                metricsFilename, metricsBase | worldModelMetrics | behaviorMetrics
            )
            plotMetrics(
                f"{metricsFilename}",
                savePath=f"{plotFilename}",
                title=f"{config.environmentName}",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="car-racing-v3.yml")
    main(parser.parse_args().config)
