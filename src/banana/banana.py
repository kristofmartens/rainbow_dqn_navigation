from unityagents import UnityEnvironment
import numpy as np
import os
import gym
import gym.spaces


class Banana(gym.Env):
    render_modes = ['training', 'realtime']

    def __init__(self):
        # Default render mode is training
        self.render_mode = Banana.render_modes[0]

        # Load the environment
        directory, _ = os.path.split(__file__)
        self.env = UnityEnvironment(os.path.join(directory, 'data', 'Banana.app'))
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # Observation space
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(self.brain.vector_observation_space_size,),
                                                dtype=np.float)

        # Action space
        self.action_space = gym.spaces.Discrete(self.brain.vector_action_space_size)

        # No need for memory

    def reset(self):
        train_mode = True if self.render_mode is 'training' else False
        env_info = self.env.reset(train_mode=train_mode)

        return env_info[self.brain_name].vector_observations[0]

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        return env_info.vector_observations[0], env_info.rewards[0], env_info.local_done[0], {}

    def render(self, mode=render_modes[0]):
        if mode not in Banana.render_modes:
            raise ValueError("Supported values are: {}".format(", ".join(Banana.render_modes)))
        self.render_mode = mode

    def close(self):
        super(Banana, self).close()

    def seed(self, seed=None):
        raise NotImplementedError
