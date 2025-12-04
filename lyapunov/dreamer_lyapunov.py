from networkx import config
import torch
import torch.nn as nn
from torch.distributions import (
    kl_divergence,
    Independent,
    OneHotCategoricalStraightThrough,
    Normal,
)
import numpy as np
import os

from lyapunov.networks_lyapunov import (
    EncoderVector,
    DecoderVector,
    RecurrentModel,
    PriorNet,
    PosteriorNet,
    RewardModel,
    ContinueModel,
    EncoderConv,
    DecoderConv,
    Actor,
    Critic,
    LyapunovModel,  # Import the new Lyapunov model
)
from utils import computeLambdaValues, Moments
from buffer import ReplayBuffer
import imageio


class DreamerLyapunov:
    def __init__(
        self, observationShape, actionSize, actionLow, actionHigh, device, config
    ):
        self.observationShape = observationShape
        self.actionSize = actionSize
        self.config = config
        self.device = device

        self.recurrentSize = config.recurrentSize
        self.latentSize = config.latentLength * config.latentClasses
        self.fullStateSize = config.recurrentSize + self.latentSize

        # get the equilibrium point if specified (Notice it is in the observation space)
        if hasattr(config, "equilibriumPoint"):
            self.equilibriumPoint = torch.tensor(
                config.equilibriumPoint, device=self.device
            ).float()
        else:
            self.equilibriumPoint = None

        self.actor = Actor(
            self.fullStateSize, actionSize, actionLow, actionHigh, device, config.actor
        ).to(self.device)
        self.critic = Critic(self.fullStateSize, config.critic).to(self.device)

        # Initialize Lyapunov function V(x)
        # Notice that we use the original observation shape here for Lyapunov
        # Meaning that we use the decoder to map back to observation space
        self.lyapunovModel = LyapunovModel(
            np.prod(observationShape), config.lyapunov
        ).to(self.device)

        # Determine observation type
        self.isImageObs = len(observationShape) == 3

        # Choose encoder/decoder based on observation type
        if self.isImageObs:
            self.encoder = EncoderConv(
                observationShape, self.config.encodedObsSize, config.encoder
            ).to(self.device)
            self.decoder = DecoderConv(
                self.fullStateSize, observationShape, config.decoder
            ).to(self.device)

        else:
            obsSize = np.prod(observationShape)
            self.encoder = EncoderVector(
                obsSize, self.config.encodedObsSize, self.config.encoder
            ).to(self.device)
            recurrentAndLatentSize = self.recurrentSize + self.latentSize
            self.decoder = DecoderVector(
                recurrentAndLatentSize, obsSize, self.config.decoder
            ).to(self.device)

        self.recurrentModel = RecurrentModel(
            config.recurrentSize, self.latentSize, actionSize, config.recurrentModel
        ).to(self.device)
        self.priorNet = PriorNet(
            config.recurrentSize,
            config.latentLength,
            config.latentClasses,
            config.priorNet,
        ).to(self.device)
        self.posteriorNet = PosteriorNet(
            config.recurrentSize + config.encodedObsSize,
            config.latentLength,
            config.latentClasses,
            config.posteriorNet,
        ).to(self.device)
        self.rewardPredictor = RewardModel(self.fullStateSize, config.reward).to(
            self.device
        )
        if config.useContinuationPrediction:
            self.continuePredictor = ContinueModel(
                self.fullStateSize, config.continuation
            ).to(self.device)

        self.buffer = ReplayBuffer(observationShape, actionSize, config.buffer, device)
        self.valueMoments = Moments(device)

        self.worldModelParameters = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.recurrentModel.parameters())
            + list(self.priorNet.parameters())
            + list(self.posteriorNet.parameters())
            + list(self.rewardPredictor.parameters())
        )
        if self.config.useContinuationPrediction:
            self.worldModelParameters += list(self.continuePredictor.parameters())

        self.worldModelOptimizer = torch.optim.Adam(
            self.worldModelParameters, lr=self.config.worldModelLR
        )
        self.actorOptimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actorLR
        )
        self.criticOptimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.criticLR
        )
        # Add Lyapunov optimizer
        self.lyapunovOptimizer = torch.optim.Adam(
            self.lyapunovModel.parameters(), lr=self.config.lyapunovLR
        )

        self.totalEpisodes = 0
        self.totalEnvSteps = 0
        self.totalGradientSteps = 0

    def worldModelTraining(self, data):
        encodedObservations = self.encoder(
            data.observations.view(-1, *self.observationShape)
        ).view(
            self.config.batchSize, self.config.batchLength, -1
        )  # collapse batch and batchLength dimensions before the forward pass and then reshape back

        previousRecurrentState = torch.zeros(
            self.config.batchSize, self.recurrentSize, device=self.device
        )  # Initialize the recurrent state to zeros
        previousLatentState = torch.zeros(
            self.config.batchSize, self.latentSize, device=self.device
        )  # Initialize the latent state to zeros

        recurrentStates, priorsLogits, posteriors, posteriorsLogits = [], [], [], []
        for t in range(1, self.config.batchLength):
            recurrentState = self.recurrentModel(
                previousRecurrentState, previousLatentState, data.actions[:, t - 1]
            )
            _, priorLogits = self.priorNet(recurrentState)
            posterior, posteriorLogits = self.posteriorNet(
                torch.cat((recurrentState, encodedObservations[:, t]), -1)
            )

            recurrentStates.append(recurrentState)
            priorsLogits.append(priorLogits)
            posteriors.append(posterior)
            posteriorsLogits.append(posteriorLogits)

            previousRecurrentState = recurrentState
            previousLatentState = posterior

        recurrentStates = torch.stack(
            recurrentStates, dim=1
        )  # (batchSize, batchLength-1, recurrentSize)
        priorsLogits = torch.stack(
            priorsLogits, dim=1
        )  # (batchSize, batchLength-1, latentLength, latentClasses)
        posteriors = torch.stack(
            posteriors, dim=1
        )  # (batchSize, batchLength-1, latentLength*latentClasses)
        posteriorsLogits = torch.stack(
            posteriorsLogits, dim=1
        )  # (batchSize, batchLength-1, latentLength, latentClasses)
        fullStates = torch.cat(
            (recurrentStates, posteriors), dim=-1
        )  # (batchSize, batchLength-1, recurrentSize + latentLength*latentClasses)

        reconstructionMeans = self.decoder(
            fullStates.view(-1, self.fullStateSize)
        ).view(
            self.config.batchSize, self.config.batchLength - 1, *self.observationShape
        )
        reconstructionDistribution = Independent(
            Normal(reconstructionMeans, 1), len(self.observationShape)
        )
        reconstructionLoss = -reconstructionDistribution.log_prob(
            data.observations[:, 1:]
        ).mean()

        rewardDistribution = self.rewardPredictor(fullStates)
        rewardLoss = -rewardDistribution.log_prob(
            data.rewards[:, 1:].squeeze(-1)
        ).mean()

        priorDistribution = Independent(
            OneHotCategoricalStraightThrough(logits=priorsLogits), 1
        )
        priorDistributionSG = Independent(
            OneHotCategoricalStraightThrough(logits=priorsLogits.detach()), 1
        )
        posteriorDistribution = Independent(
            OneHotCategoricalStraightThrough(logits=posteriorsLogits), 1
        )
        posteriorDistributionSG = Independent(
            OneHotCategoricalStraightThrough(logits=posteriorsLogits.detach()), 1
        )

        priorLoss = kl_divergence(posteriorDistributionSG, priorDistribution)
        posteriorLoss = kl_divergence(posteriorDistribution, priorDistributionSG)
        freeNats = torch.full_like(priorLoss, self.config.freeNats)

        priorLoss = torch.maximum(priorLoss, freeNats).mean()
        posteriorLoss = torch.maximum(posteriorLoss, freeNats).mean()

        continueLoss = (
            -self.continuePredictor(fullStates)
            .log_prob(data.dones[:, 1:].squeeze(-1).logical_not())
            .mean()
            if self.config.useContinuationPrediction
            else torch.tensor(0, device=self.device)
        )

        loss = (
            reconstructionLoss
            + rewardLoss
            + self.config.betaPrior * priorLoss
            + self.config.betaPosterior * posteriorLoss
            + continueLoss
        )

        self.worldModelOptimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(
            self.worldModelParameters,
            self.config.gradientClip,
            norm_type=self.config.gradientNormType,
        )
        self.worldModelOptimizer.step()

        initialStates = torch.cat(
            (recurrentStates[:, 0], posteriors[:, 0]), -1
        ).detach()

        metrics = {
            "reconstructionLoss": reconstructionLoss.item(),
            "rewardLoss": rewardLoss.item(),
            "priorLoss": priorLoss.item(),
            "posteriorLoss": posteriorLoss.item(),
            "continueLoss": (
                continueLoss.item() if self.config.useContinuationPrediction else 0
            ),
            "totalLoss": loss.item(),
        }
        return initialStates, metrics

    def behaviorTraining(self, initialStates):
        """Modified behavior training with Lyapunov regularization."""
        recurrentState = initialStates[:, : self.recurrentSize]
        latentState = initialStates[:, self.recurrentSize :]

        fullStates = [torch.cat((recurrentState, latentState), -1)]
        actions = []
        logprobs = []
        entropies = []
        lyapunovValues = []  # Store Lyapunov values for regularization

        for _ in range(self.config.imaginationHorizon):
            action, logprob, entropy = self.actor(fullStates[-1], training=True)
            recurrentState = self.recurrentModel(recurrentState, latentState, action)
            latentState, _ = self.priorNet(recurrentState)

            # Compute Lyapunov value for current state
            reconstructedObservation = self.decoder(fullStates[-1])
            lyapunovValue = self.lyapunovModel(reconstructedObservation)

            fullState = torch.cat((recurrentState, latentState), -1)

            fullStates.append(fullState)
            actions.append(action)
            logprobs.append(logprob)
            entropies.append(entropy)
            lyapunovValues.append(lyapunovValue)

        fullStates = torch.stack(
            fullStates, dim=1
        )  # (batchSize*batchLength, imaginationHorizon+1, fullStateSize)
        logprobs = torch.stack(
            logprobs, dim=1
        )  # (batchSize*batchLength, imaginationHorizon)
        entropies = torch.stack(
            entropies, dim=1
        )  # (batchSize*batchLength, imaginationHorizon)
        lyapunovValues = torch.stack(
            lyapunovValues, dim=1
        )  # (batchSize*batchLength, imaginationHorizon)

        # Standard Dreamer components
        predictedRewards = self.rewardPredictor(fullStates[:, :-1]).mean
        values = self.critic(fullStates).mean
        continues = (
            self.continuePredictor(fullStates).mean
            if self.config.useContinuationPrediction
            else torch.full_like(predictedRewards, self.config.discount)
        )
        lambdaValues = computeLambdaValues(
            predictedRewards, values, continues, self.config.lambda_
        )

        _, inverseScale = self.valueMoments(lambdaValues)
        advantages = (lambdaValues - values[:, :-1]) / inverseScale

        # Compute Lyapunov decrease penalty: we want V(x_{t+1}) - V(x_t) < 0
        # So we penalize positive differences
        lyapunovDifferences = lyapunovValues[:, 1:] - lyapunovValues[:, :-1]

        # Option 1: Soft constraint - penalize when V increases
        lyapunovPenalty = torch.relu(lyapunovDifferences).mean()

        # Modified actor loss with Lyapunov regularization
        actorLoss = (
            -torch.mean(
                advantages.detach() * logprobs + self.config.entropyScale * entropies
            )
            + self.config.lyapunovLambda * lyapunovPenalty
        )

        self.actorOptimizer.zero_grad()
        actorLoss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.gradientClip,
            norm_type=self.config.gradientNormType,
        )
        self.actorOptimizer.step()

        # Update critic (unchanged)
        valueDistributions = self.critic(fullStates[:, :-1].detach())
        criticLoss = -torch.mean(valueDistributions.log_prob(lambdaValues.detach()))

        self.criticOptimizer.zero_grad()
        criticLoss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.gradientClip,
            norm_type=self.config.gradientNormType,
        )
        self.criticOptimizer.step()

        # Train Lyapunov with proper objectives:
        # 1. Encourage V to decrease along trajectories
        LyapunovDecayLoss = lyapunovDifferences.mean()  # Should be negative

        # 2. Ensure positive definiteness (V >= 0)
        PositiveLoss = torch.relu(-lyapunovValues).mean()

        # 3. Zero at equilibrium (only if specified)
        if self.equilibriumPoint is not None:
            lyapunovAtEquilibrium = self.lyapunovModel(
                self.equilibriumPoint.unsqueeze(0)
            )
            LyapunovEquilibriumLoss = lyapunovAtEquilibrium.pow(2).mean()
        else:
            LyapunovEquilibriumLoss = 0.0

        lyapunovLoss = (
            LyapunovDecayLoss + 0.1 * PositiveLoss + 0.1 * LyapunovEquilibriumLoss
        )

        self.lyapunovOptimizer.zero_grad()
        lyapunovLoss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(
            self.lyapunovModel.parameters(),
            self.config.gradientClip,
            norm_type=self.config.gradientNormType,
        )
        self.lyapunovOptimizer.step()

        metrics = {
            "actorLoss": actorLoss.item(),
            "criticLoss": criticLoss.item(),
            "lyapunovLoss": lyapunovLoss.item(),
            "lyapunovPenalty": lyapunovPenalty.item(),
            "lyapunovMean": lyapunovValues.mean().item(),
            "lyapunovStd": lyapunovValues.std().item(),
            "entropies": entropies.mean().item(),
            "logprobs": logprobs.mean().item(),
            "advantages": advantages.mean().item(),
            "criticValues": values.mean().item(),
        }
        return metrics

    @torch.no_grad()
    def environmentInteraction(
        self,
        env,
        numEpisodes,
        seed=None,
        evaluation=False,
        saveVideo=False,
        filename="videos/unnamedVideo",
        fps=30,
        macroBlockSize=16,
    ):
        scores = []
        lyapunovTrajectories = []  # Store Lyapunov values for analysis

        for i in range(numEpisodes):
            recurrentState, latentState = torch.zeros(
                1, self.recurrentSize, device=self.device
            ), torch.zeros(1, self.latentSize, device=self.device)
            action = torch.zeros(1, self.actionSize).to(self.device)

            observation = env.reset(seed=(seed + self.totalEpisodes if seed else None))
            encodedObservation = self.encoder(
                torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            )

            currentScore, stepCount, done, frames = 0, 0, False, []
            episodeLyapunovValues = []

            while not done:
                recurrentState = self.recurrentModel(
                    recurrentState, latentState, action
                )
                latentState, _ = self.posteriorNet(
                    torch.cat((recurrentState, encodedObservation.view(1, -1)), -1)
                )
                fullState = torch.cat((recurrentState, latentState), -1)

                # Track Lyapunov value during evaluation
                if evaluation:
                    # Use the decoder to reconstruct observation from fullState
                    reconstructedObservation = self.decoder(fullState)
                    lyapunovValue = self.lyapunovModel(reconstructedObservation)
                    episodeLyapunovValues.append(lyapunovValue.item())

                action = self.actor(fullState)
                actionNumpy = action.cpu().numpy().reshape(-1)

                nextObservation, reward, done = env.step(actionNumpy)
                if not evaluation:
                    self.buffer.add(
                        observation, actionNumpy, reward, nextObservation, done
                    )

                if saveVideo and i == 0:
                    frame = env.render()
                    targetHeight = (
                        (frame.shape[0] + macroBlockSize - 1)
                        // macroBlockSize
                        * macroBlockSize
                    )
                    targetWidth = (
                        (frame.shape[1] + macroBlockSize - 1)
                        // macroBlockSize
                        * macroBlockSize
                    )
                    frames.append(
                        np.pad(
                            frame,
                            (
                                (0, targetHeight - frame.shape[0]),
                                (0, targetWidth - frame.shape[1]),
                                (0, 0),
                            ),
                            mode="edge",
                        )
                    )

                encodedObservation = self.encoder(
                    torch.from_numpy(nextObservation)
                    .float()
                    .unsqueeze(0)
                    .to(self.device)
                )
                observation = nextObservation

                currentScore += reward
                stepCount += 1
                if done:
                    scores.append(currentScore)
                    if evaluation:
                        lyapunovTrajectories.append(episodeLyapunovValues)

                    if not evaluation:
                        self.totalEpisodes += 1
                        self.totalEnvSteps += stepCount

                    if saveVideo and i == 0:
                        finalFilename = f"{filename}_reward_{currentScore:.0f}.mp4"
                        with imageio.get_writer(finalFilename, fps=fps) as video:
                            for frame in frames:
                                video.append_data(frame)
                    break

        # Compute Lyapunov stability metrics for evaluation
        if evaluation and lyapunovTrajectories:
            # Check monotonicity: percentage of steps where V decreases
            monotonicSteps = 0
            totalSteps = 0
            for trajectory in lyapunovTrajectories:
                for i in range(1, len(trajectory)):
                    if trajectory[i] < trajectory[i - 1]:
                        monotonicSteps += 1
                    totalSteps += 1
            monotonicityRate = monotonicSteps / totalSteps if totalSteps > 0 else 0

            print(f"Lyapunov monotonicity rate: {monotonicityRate:.2%}")

        return sum(scores) / numEpisodes if numEpisodes else None

    def saveCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith(".pth"):
            checkpointPath += ".pth"

        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "recurrentModel": self.recurrentModel.state_dict(),
            "priorNet": self.priorNet.state_dict(),
            "posteriorNet": self.posteriorNet.state_dict(),
            "rewardPredictor": self.rewardPredictor.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "lyapunovModel": self.lyapunovModel.state_dict(),  # Save Lyapunov model
            "worldModelOptimizer": self.worldModelOptimizer.state_dict(),
            "criticOptimizer": self.criticOptimizer.state_dict(),
            "actorOptimizer": self.actorOptimizer.state_dict(),
            "lyapunovOptimizer": self.lyapunovOptimizer.state_dict(),  # Save Lyapunov optimizer
            "totalEpisodes": self.totalEpisodes,
            "totalEnvSteps": self.totalEnvSteps,
            "totalGradientSteps": self.totalGradientSteps,
        }
        if self.config.useContinuationPrediction:
            checkpoint["continuePredictor"] = self.continuePredictor.state_dict()
        torch.save(checkpoint, checkpointPath)

    def loadCheckpoint(self, checkpointPath):
        if not checkpointPath.endswith(".pth"):
            checkpointPath += ".pth"
        if not os.path.exists(checkpointPath):
            raise FileNotFoundError(f"Checkpoint file not found at: {checkpointPath}")

        checkpoint = torch.load(checkpointPath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.recurrentModel.load_state_dict(checkpoint["recurrentModel"])
        self.priorNet.load_state_dict(checkpoint["priorNet"])
        self.posteriorNet.load_state_dict(checkpoint["posteriorNet"])
        self.rewardPredictor.load_state_dict(checkpoint["rewardPredictor"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

        # Load Lyapunov model if it exists
        if "lyapunovModel" in checkpoint:
            self.lyapunovModel.load_state_dict(checkpoint["lyapunovModel"])

        self.worldModelOptimizer.load_state_dict(checkpoint["worldModelOptimizer"])
        self.criticOptimizer.load_state_dict(checkpoint["criticOptimizer"])
        self.actorOptimizer.load_state_dict(checkpoint["actorOptimizer"])

        # Load Lyapunov optimizer if it exists
        if "lyapunovOptimizer" in checkpoint:
            self.lyapunovOptimizer.load_state_dict(checkpoint["lyapunovOptimizer"])

        self.totalEpisodes = checkpoint["totalEpisodes"]
        self.totalEnvSteps = checkpoint["totalEnvSteps"]
        self.totalGradientSteps = checkpoint["totalGradientSteps"]
        if self.config.useContinuationPrediction:
            self.continuePredictor.load_state_dict(checkpoint["continuePredictor"])
