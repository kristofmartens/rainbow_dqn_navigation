from model import QNetwork
import rainbow_dqn_agent
import torch
from torch.nn import functional as F
from torch.optim import Adam
import random


class DQNAgent:
    def __init__(self, env, gamma=0.99, device='cpu', batch_size=32, learning_rate=1e-4):
        self.env = env
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.online_net = QNetwork(self.env).to(device=device)
        self.online_net.train()

        self.target_net = QNetwork(self.env).to(device=device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = Adam(self.online_net.parameters(), lr=learning_rate)

    def act(self, state):
        with torch.no_grad():
            return self.online_net(state).max(1)[1].item()

    def act_e_greedy(self, state, eps=0.001):
        if random.random() < eps:
            return self.act(state)
        else:
            return self.env.action_space.sample()

    def learn(self, memory):
        # Draw set of samples
        samples = memory.sample(self.batch_size)
        batch = rainbow_dqn_agent.Transition(*zip(*samples))  # Converts to transition of batch arrays

        # with torch.no_grad():
        state_batch = torch.cat(batch.state)
        action_batch = torch.LongTensor(batch.action).view(-1, 1)
        reward_batch = torch.Tensor(batch.reward).view(-1, 1)
        next_state_batch = torch.cat(batch.next_state)
        not_done_batch = torch.Tensor([not i for i in batch.done]).view(-1, 1)

        # Compute the Q values for this state and chosen actions
        state_action_values = self.online_net(state_batch).gather(1, action_batch)
        next_state_values = self.online_net(next_state_batch).max(1)[0].view(-1, 1)
        expected_state_action_values = next_state_values * self.gamma * not_done_batch + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize model
        with torch.enable_grad():
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def q(self, state):
        with torch.no_grad():
            return self.online_net(state)

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
