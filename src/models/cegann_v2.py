from collections.abc import Callable

import torch
from torch.nn import Module, ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import Linear
from torch_geometric.utils import trim_to_layer

from src.models.layers.embedding import GeometricEmbedding
from src.models.layers.geo_conv import GeometricConv
from src.models.layers.readout import BondToAtomReadout


# TODO we can probably trim down some arguments
# such as hidden channels in conv and embedding output channels
class CEGANNv2(Module):
    """CEGANNv2 model for node classification on crystal graphs.

    Args:
        num_classes: Number of target classes.
        emb_num_radial: Number of radial basis functions for distance encoding.
        emb_num_angular: Number of angular basis functions for angle encoding.
        emb_node_out_channels: Output channels for node embeddings.
        emb_edge_out_channels: Output channels for edge embeddings.
        emb_num_layers: Number of layers in the embedding module.
        emb_hidden_channels: Hidden channels in the embedding module.
        conv_hidden_channels: Hidden channels in the convolutional layers.
        conv_node_out_channels: Output channels for node features in the convolutional layers.
        conv_edge_out_channels: Output channels for edge features in the convolutional layers.
        conv_num_layers: Number of convolutional layers.
        conv_heads: Number of attention heads in the convolutional layers.
        conv_norm: Normalization method in the convolutional layers.
        dropout: Dropout rate.
        act: Activation function.
    """

    def __init__(
        self,
        num_classes: int,
        emb_num_radial: int,
        emb_num_angular: int,
        emb_node_out_channels: int,
        emb_edge_out_channels: int,
        emb_num_layers: int = 1,
        emb_hidden_channels: int | None = None,
        conv_hidden_channels: int = 128,
        conv_node_out_channels: int = 128,
        conv_edge_out_channels: int = 128,
        conv_num_layers: int = 2,
        conv_heads: int = 1,
        conv_norm: str | Callable | None = "layernorm",
        dropout: float = 0.1,
        act: str | Callable | None = "silu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.embedding = GeometricEmbedding(
            num_radial=emb_num_radial,
            num_angular=emb_num_angular,
            node_out_channels=emb_node_out_channels,
            edge_out_channels=emb_edge_out_channels,
            num_layers=emb_num_layers,
            hidden_channels=emb_hidden_channels,
            act=act,
        )

        node_channels = emb_node_out_channels
        edge_channels = emb_edge_out_channels

        self.convs = ModuleList()
        for _ in range(conv_num_layers - 2):
            self.convs.append(
                GeometricConv(
                    node_in_channels=node_channels,
                    edge_in_channels=edge_channels,
                    hidden_channels=conv_hidden_channels,
                    node_out_channels=conv_hidden_channels,
                    edge_out_channels=conv_hidden_channels,
                    heads=conv_heads,
                    dropout=dropout,
                    norm=conv_norm,
                    act=act,
                    **kwargs,
                )
            )
            node_channels = conv_hidden_channels
            edge_channels = conv_hidden_channels

        self.convs.append(
            GeometricConv(
                node_in_channels=node_channels,
                edge_in_channels=edge_channels,
                hidden_channels=conv_hidden_channels,
                node_out_channels=conv_node_out_channels,
                edge_out_channels=conv_edge_out_channels,
                heads=conv_heads,
                dropout=dropout,
                norm=conv_norm,
                act=act,
                **kwargs,
            )
        )

        self.block_dropout = torch.nn.Dropout(p=dropout)
        self.readout = BondToAtomReadout(reduce="mean", incidence="out")
        self.out_head = Linear(-1, num_classes, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset model parameters."""
        self.embedding.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.out_head.reset_parameters()

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the model."""
        assert data.x is not None, "Node features are required."
        assert data.edge_attr is not None, "Edge attributes are required."
        assert data.edge_index is not None, "Edge index is required."

        # Extract data
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Encode distances and angles
        h_bond, h_angle = self.embedding(x, edge_attr)

        # Convolution blocks on the line graph
        for i, conv in enumerate(self.convs):
            # Trim to sampled nodes/edges if neighbor sampling is used
            if hasattr(data, "num_sampled_nodes_per_hop"):
                num_sampled_nodes_per_hop: list[int] = data.num_sampled_nodes_per_hop
                num_sampled_edges_per_hop: list[int] = data.num_sampled_edges_per_hop
                x, edge_index, edge_attr = trim_to_layer(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_attr,
                )

            x, edge_attr = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            edge_attr = self.block_dropout(edge_attr)
            x = self.block_dropout(x)

        # Pooling from bonds to atoms
        bond_source = data.bond_source if hasattr(data, "bond_source") else None
        num_atoms = data.num_atoms if hasattr(data, "num_atoms") else None
        h_atom = self.readout(
            h_bond,
            num_atoms,
            bond_source=bond_source,
        )

        # Final MLP for node classification
        out = self.out_head(h_atom)

        return out
