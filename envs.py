import gymnasium as gym
import numpy as np


def getEnvProperties(env):
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "Sorry, supporting only continuous action space for now"
    observationShape = env.observation_space.shape
    actionSize = env.action_space.shape[0]
    actionLow = env.action_space.low.tolist()
    actionHigh = env.action_space.high.tolist()
    return observationShape, actionSize, actionLow, actionHigh


class GymPixelsProcessingWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        observationSpace = self.observation_space
        newObsShape = observationSpace.shape[-1:] + observationSpace.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=newObsShape, dtype=np.float32
        )

    def observation(self, observation):
        observation = np.transpose(observation, (2, 0, 1)) / 255.0
        return observation


class CleanGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs


import dm_control.suite as suite
from dm_control import suite


class DMControlWrapper(gym.Env):
    def __init__(self, domain, task):
        self.env = suite.load(domain_name=domain, task_name=task)
        spec = self.env.observation_spec()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(sum([np.prod(v.shape) for v in spec.values()]),),
            dtype=np.float32,
        )
        spec = self.env.action_spec()
        self.action_space = gym.spaces.Box(
            low=spec.minimum, high=spec.maximum, dtype=np.float32
        )
        self._last_time_step = None  # Store for rendering

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        time_step = self.env.reset()
        self._last_time_step = time_step
        return self._get_obs(time_step), {}

    def step(self, action):
        time_step = self.env.step(action)
        self._last_time_step = time_step
        obs = self._get_obs(time_step)
        reward = time_step.reward or 0
        done = time_step.last()
        return obs, reward, done, False, {}

    def render(self):
        # Render to RGB array
        return self.env.physics.render(camera_id=0, height=240, width=320)

    def _get_obs(self, time_step):
        obs_list = []
        for v in time_step.observation.values():
            obs_list.append(v.flatten())
        return np.concatenate(obs_list, axis=0).astype(np.float32)
