import gym
import numpy as np

# For discrete observation spaces (e.g., CliffWalking),
# return observations of type np.array rather than int
class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.discrete = isinstance(env.observation_space, gym.spaces.Discrete)
        print(f'ObservationWrapper::__init__: discrete={self.discrete}')

    def observation(self, obs):
        assert not self.discrete or isinstance(obs, (np.int64, int))
        return np.array([obs]) if self.discrete else obs

# For discrete action spaces (e.g., CliffWalking),
# let step() accept actions of type np.array
class ActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        print(f'ActionWrapper::__init__: discrete={self.discrete}')

    def step(self, a):
        if self.discrete and isinstance(a, np.ndarray):
            return self.env.step(a.item())
        else:
            return self.env.step(a)

