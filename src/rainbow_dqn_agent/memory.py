import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'episode_counter'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.episode_counter = 0

    def append(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done, self.episode_counter))
        self.episode_counter = self.episode_counter + 1 if not done else 0

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

