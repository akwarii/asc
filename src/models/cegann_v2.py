from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import Linear

from src.models.base import BaseModel
from src.models.layers.embedding import GeometricEmbedding
from src.models.layers.geo_conv import GeometricConv
from src.models.layers.readout import BondToAtomReadout


class CEGANNv2(BaseModel):
    """CEGANNv2 model for node classification on crystal graphs.

    Args:
        out_channels: Number of target classes.
        emb_num_radial: Number of radial basis functions for distance encoding.
        emb_num_angular: Number of angular basis functions for angle encoding.
        emb_num_channels: Output channels for node embeddings.
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
        out_channels: int,
        emb_num_radial: int = 16,
        emb_num_angular: int = 16,
        emb_num_channels: int = 128,
        emb_num_layers: int = 2,
        conv_hidden_channels: int = 128,
        conv_node_out_channels: int = 128,
        conv_edge_out_channels: int = 128,
        conv_num_layers: int = 2,
        conv_heads: int = 1,
        conv_concat: bool = True,
        conv_residual: bool = True,
        conv_norm: str | Callable | None = "layernorm",
        dropout: float = 0.1,
        act: str | Callable | None = "silu",
        **kwargs,
    ) -> None:
        super().__init__()

        self.embedding = GeometricEmbedding(
            num_radial=emb_num_radial,
            num_angular=emb_num_angular,
            num_channels=emb_num_channels,
            num_layers=emb_num_layers,
            act=act,
        )

        node_in, edge_in = emb_num_channels, emb_num_channels

        self.convs = nn.ModuleList()
        for layer in range(conv_num_layers):
            is_last = layer == conv_num_layers - 1
            node_out = conv_node_out_channels if is_last else conv_hidden_channels
            edge_out = conv_edge_out_channels if is_last else conv_hidden_channels

            self.convs.append(
                GeometricConv(
                    node_in_channels=node_in,
                    edge_in_channels=edge_in,
                    hidden_channels=conv_hidden_channels,
                    node_out_channels=node_out,
                    edge_out_channels=edge_out,
                    heads=conv_heads,
                    concat=conv_concat if not is_last else False,
                    dropout=dropout,
                    norm=conv_norm,
                    residual=conv_residual,
                    act=act,
                    **kwargs,
                )
            )
            node_in, edge_in = node_out, edge_out

        self.readout = BondToAtomReadout(reduce="mean", incidence="out")
        self.out_head = Linear(node_in, out_channels, bias=False)
        self.out_channels = out_channels

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset model parameters."""
        self.embedding.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.out_head.reset_parameters()

    @property
    def num_layers(self) -> int:
        """Number of convolutional layers."""
        return len(self.convs)

    @property
    def device(self) -> torch.device:
        """Device on which the model is located."""
        return next(self.parameters()).device

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        bond_source: Tensor | None = None,
        num_atoms: int | Tensor | None = None,
        num_sampled_nodes_per_hop: list[int] | None = None,
        num_sampled_edges_per_hop: list[int] | None = None,
    ) -> Tensor:
        """Forward pass of the model."""
        assert edge_attr is not None, "edge_attr cannot be None for CEGANNv2"
        assert bond_source is not None, "bond_source cannot be None for CEGANNv2"
        assert num_atoms is not None, "num_atoms cannot be None for CEGANNv2"

        if num_sampled_nodes_per_hop is not None:
            raise NotImplementedError("Neighbor sampling is not implemented yet for CEGANNv2.")

        # Encode distances and angles
        x, edge_attr = self.embedding(x, edge_attr)

        for conv in self.convs:
            x, edge_attr = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # During batching, num_atoms can be a tensor
        # Keep as tensor to avoid graph breaks in torch.compile
        if isinstance(num_atoms, Tensor):
            num_atoms = num_atoms.sum()

        h_atom = self.readout(x, num_atoms, bond_source=bond_source)
        out = self.out_head(h_atom)

        return out
