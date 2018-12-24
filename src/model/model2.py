from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    def __init__(self, env, hidden_layers=None):
        super(QNetwork, self).__init__()

        if hidden_layers is None:
            hidden_layers = [64, 64, 64]

        layer_dimensions = [env.observation_space.shape[0]] + hidden_layers + [env.action_space.n]

        self.fc_1 = nn.Linear(layer_dimensions[0], layer_dimensions[1])
        self.fc_bn_1 = nn.BatchNorm1d(layer_dimensions[1])

        self.fc_2 = nn.Linear(layer_dimensions[1], layer_dimensions[2])
        self.fc_bn_2 = nn.BatchNorm1d(layer_dimensions[2])

        self.fc_3 = nn.Linear(layer_dimensions[2], layer_dimensions[3])
        self.fc_bn_3 = nn.BatchNorm1d(layer_dimensions[3])

        self.fc_4 = nn.Linear(layer_dimensions[3], layer_dimensions[4])

    def forward(self, x):
        # Add hidden layers
        x = self.fc_bn_1(F.relu(self.fc_1(x)))
        x = self.fc_bn_2(F.relu(self.fc_2(x)))
        x = self.fc_bn_3(F.relu(self.fc_3(x)))
        x = self.fc_4(x)

        return x
