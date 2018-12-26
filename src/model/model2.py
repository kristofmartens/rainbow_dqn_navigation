from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    def __init__(self, env, hidden_layers=None):
        super(QNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64, 64]

        # Can't create NN in a loop, or it can't find the variables to optimize
        layer_dimensions = [env.observation_space.shape[0]] + hidden_layers + [env.action_space.n]

        # Linear layers
        self.fc_1 = nn.Linear(layer_dimensions[0], layer_dimensions[1])
        self.fc_bn_1 = nn.BatchNorm1d(layer_dimensions[1])

        self.fc_2 = nn.Linear(layer_dimensions[1], layer_dimensions[2])
        self.fc_bn_2 = nn.BatchNorm1d(layer_dimensions[2])

        self.fc_3 = nn.Linear(layer_dimensions[2], layer_dimensions[3])
        self.fc_bn_3 = nn.BatchNorm1d(layer_dimensions[3])

        # Split up between value and advantage Dueling DQN
        self.fc_h_v = nn.Linear(layer_dimensions[3], layer_dimensions[3])
        self.fc_h_a = nn.Linear(layer_dimensions[3], layer_dimensions[3])

        self.fc_z_v = nn.Linear(layer_dimensions[3], 1)
        self.fc_z_a = nn.Linear(layer_dimensions[3], layer_dimensions[4])

        self.fc_4 = nn.Linear(layer_dimensions[3], layer_dimensions[4])

    def forward(self, x):
        # Add hidden layers
        x = self.fc_bn_1(F.relu(self.fc_1(x)))
        x = self.fc_bn_2(F.relu(self.fc_2(x)))
        x = self.fc_bn_3(F.relu(self.fc_3(x)))

        value_stream = self.fc_z_v(F.relu(self.fc_h_v(x))).view(-1, 1)
        advantage_stream = self.fc_z_a(F.relu(self.fc_h_a(x)))
        x = value_stream + advantage_stream - advantage_stream.mean(1, keepdim=True)

        return x
