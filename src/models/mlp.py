from torch import nn
from torch_geometric.nn import Linear


class MLP(nn.Module):
    """MLP.

    Args:
        c_in: Dimension of input features
        c_hidden: Dimension of hidden features
        c_out: Dimension of the output features. Usually number of classes in classification
        num_layers: Number of hidden layers
        dp_rate: Dropout rate to apply throughout the network
    """

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, dp_rate=0.1):
        super().__init__()
        layers = []
        in_channels, out_channels = c_in, c_hidden
        for _ in range(num_layers - 1):
            layers += [
                Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate),
            ]
            in_channels = c_hidden
        layers += [Linear(in_channels, c_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward.

        Args:
            x: Input features per node
        """
        return self.layers(x.edge_dist)
