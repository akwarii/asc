from math import sqrt

import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter


class MLP(torch.nn.Module):
    """Multi-Layer perceptron."""

    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            self.layers.append(
                torch.nn.Linear(
                    input_size if i == 0 else hidden_size,
                    output_size if i == layers - 1 else hidden_size,
                )
            )
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
        if layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class EGAT(MessagePassing):
    def __init__(
        self,
        in_node_channels,
        in_edge_channels,
        out_channels,
        heads=3,
        layers=3,
        bias=True,
        get_attn=False,
        use_F=True,
        negative_slope=0.2,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_node_channels = in_node_channels
        self.in_edge_channels = in_edge_channels
        self.out_channels = out_channels
        self.heads = heads
        self.get_attn = get_attn
        self.use_F = use_F
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.node_out: torch.Tensor | None = None
        self.edge_out: torch.Tensor | None = None
        self.attn_weights = None

        # linear transformation layers for node and edge features
        self.lin_node = torch.nn.Linear(self.in_node_channels, self.out_channels, bias=True)
        self.lin_edge = torch.nn.Linear(self.in_edge_channels, self.out_channels, bias=True)
        self.lin_node_i = torch.nn.Linear(
            self.in_node_channels, self.heads * self.out_channels, bias=False
        )
        self.lin_node_j = torch.nn.Linear(
            self.in_node_channels, self.heads * self.out_channels, bias=False
        )
        self.lin_edge_ij = torch.nn.Linear(
            self.in_edge_channels, self.heads * self.out_channels, bias=False
        )

        # attention MLP to multiply with transformed node and edge features
        self.attn_A = MLP(
            3 * self.heads * self.out_channels,
            self.heads * self.out_channels,
            self.heads * self.out_channels,
            layers,
        )

        # attention layer to multiply with new edge feature to get unnormalized attention weights
        self.attn_F = torch.nn.Parameter(
            torch.FloatTensor(size=(1, self.heads, self.out_channels))
        )

        # MLPS for compressing multi-head node and edge features
        self.node_mlp = MLP(
            self.heads * self.out_channels, self.out_channels, self.out_channels, layers
        )
        self.edge_mlp = MLP(
            self.heads * self.out_channels, self.out_channels, self.out_channels, layers
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin_node.weight)
        torch.nn.init.xavier_uniform_(self.lin_edge.weight)
        torch.nn.init.xavier_uniform_(self.lin_node_i.weight)
        torch.nn.init.xavier_uniform_(self.lin_node_j.weight)
        torch.nn.init.xavier_uniform_(self.lin_edge_ij.weight)
        torch.nn.init.xavier_uniform_(self.attn_F)

    def forward(self, h, edge_index, edge_feature, size=None):
        H, C = self.heads, self.out_channels
        h_prime_i = self.lin_node_i(h)  # shape [N,H*C]
        h_prime_j = self.lin_node_j(h)  # shape [N,H*C]
        f_ij = self.lin_edge_ij(edge_feature)  # shape [E,H*C]

        # new multi-head node features
        node_out = self.propagate(edge_index, x=(h_prime_i, h_prime_j), size=size, f_ij=f_ij)
        self.node_out = self.node_mlp(node_out.reshape(-1, H * C))

        self.edge_out = self.lin_edge(edge_feature) + self.edge_mlp(
            self.edge_out.reshape(-1, H * C)
        )
        self.node_out = self.lin_node(h) + self.node_out

        if self.get_attn:
            return self.node_out, self.edge_out, self.attn_weights
        else:
            return self.node_out, self.edge_out

    def message(self, x_i, x_j, index, ptr, size_i, f_ij):
        f_prime_ij = torch.cat([x_i, f_ij, x_j], dim=-1)  # shape [E,H*C]
        f_prime_ij = self.attn_A(f_prime_ij)
        f_prime_ij = F.leaky_relu(f_prime_ij, negative_slope=self.negative_slope).reshape(
            -1, self.heads, self.out_channels
        )
        self.edge_out = f_prime_ij  # new multi-head edge features
        eps = (f_prime_ij * self.attn_F) if self.use_F else f_prime_ij
        eps = eps.sum(dim=-1).unsqueeze(-1)  # unnormalized attention weights
        alpha = softmax(eps, index, ptr, size_i)  # normalized attention weights
        alpha = F.dropout(alpha, p=self.dropout)
        self.attn_weights = alpha  # shape [E,H,1]
        out = x_j.reshape(-1, self.heads, self.out_channels) * alpha
        return out

    def aggregate(self, inputs, index, dim_size=None):
        out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return out
