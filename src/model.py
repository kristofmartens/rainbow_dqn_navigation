import torch
import torch.nn as nn
from rainbow_dqn_agent import NoisyLinear


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layers=None, std_init=0.4):
        super(QNetwork, self).__init__()
        if hidden_layers is None:
            self.hidden_layers = [64, 64]
        else:
            self.hidden_layers = hidden_layers
        self.seed = torch.manual_seed(seed)

        self.fc1 = NoisyLinear(state_size, self.hidden_layers[0], std_init=std_init)
        self.relu1 = nn.ReLU()
        self.fc2 = NoisyLinear(hidden_layers[0], self.hidden_layers[1], std_init=std_init)
        self.relu2 = nn.ReLU()
        self.fc3 = NoisyLinear(hidden_layers[1], action_size, std_init=std_init)

    def forward(self, state):
        state = self.relu1(self.fc1(state))
        state = self.relu2(self.fc2(state))
        state = self.fc3(state)
        return state

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()
