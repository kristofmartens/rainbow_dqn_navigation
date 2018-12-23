import torch
import torch.nn as nn
from torch.nn import functional as F
from rainbow_dqn_agent import NoisyLinear


class QNetwork(nn.Module):
    def __init__(self, env, seed=0, hidden_layers=None, value_dist_size=51, std_init=0.4):
        super(QNetwork, self).__init__()
        self.env = env

        if hidden_layers is None:
            self.hidden_layers = [64, 64, 64]
        else:
            self.hidden_layers = hidden_layers
        self.seed = torch.manual_seed(seed)

        self.value_dist_size = value_dist_size

        self.fc1 = NoisyLinear(self.env.observation_space.shape[0], self.hidden_layers[0], std_init=std_init)
        self.fc2 = NoisyLinear(self.hidden_layers[0], self.hidden_layers[1], std_init=std_init)

        self.fc_h_v = NoisyLinear(self.hidden_layers[1], self.hidden_layers[2], std_init=std_init)
        self.fc_h_a = NoisyLinear(self.hidden_layers[1], self.hidden_layers[2], std_init=std_init)

        self.fc_z_v = NoisyLinear(self.hidden_layers[2], self.value_dist_size, std_init=std_init)
        self.fc_z_a = NoisyLinear(self.hidden_layers[2], self.value_dist_size * self.env.action_space.n, std_init=std_init)

    def forward(self, x, log=False):
        x = x.view(-1, self.env.observation_space.shape[0])
        # 2 Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Calulate value and advantage streams
        value_stream = self.fc_z_v(F.relu(self.fc_h_v(x)))
        advantage_stream = self.fc_z_a(F.relu(self.fc_h_a(x)))

        value_stream = value_stream.view(-1, 1, self.value_dist_size)
        advantage_stream = advantage_stream.view(-1, self.env.action_space.n, self.value_dist_size)

        # Calculate Q from value and advantage streamsoit push
        q = value_stream + advantage_stream - advantage_stream.mean(1, keepdim=True)

        # TODO: was dim 2 before
        if log:
            q = F.log_softmax(q, dim=2)
        else:
            q = F.softmax(q, dim=2)

        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()
