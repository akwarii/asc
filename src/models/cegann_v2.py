import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from src.models.layers.embedding import GeoEmbedding
from src.models.layers.geo_conv import GeoConvBlock


class CEGANNv2(nn.Module):
    """CEGANNv2 model for node classification on crystal graphs.

    Model architecture:
    - GeoEmbedding layer to encode distances and angles.
    - Multiple GeoConvBlocks to perform message passing on the line graph.
    - Pooling from bond embeddings to atom embeddings.
    - Optional normalization layer on atom features.
    - Final MLP for node classification.

    Args:
        hidden_channels (int): Number of hidden channels.
        num_classes (int): Number of output classes.
        num_layers (int, optional): Number of GeoConvBlocks. Default is 3.
        heads (int, optional): Number of attention heads in GeoConvBlocks. Default is 4.
        dropout (float, optional): Dropout rate. Default is 0.1.
        concat_heads (bool, optional): Whether to concatenate heads in GeoConvBlocks.
            Default is True.
        atom_norm (str | Callable | None, optional): Normalization method for atom features.
            Default is None.
        atom_norm_kwargs (dict | None, optional): Additional arguments for atom normalization.
            Default is None.
        n_rbf_bond (int, optional): Number of radial basis functions for bond distances.
            Default is 8.
        n_rbf_angle (int, optional): Number of radial basis functions for angles. Default is 32.
    """
    def __init__(
        self,
        num_classes: int,
        *,
        # Embedding parameters
        n_rbf_bond: int = 8,
        n_rbf_angle: int = 32,
        emb_dim: int = 64,
        emb_layers: int = 1,
        emb_act: str | None = None,
        emb_hidden: int | None = None,
        emb_bias: bool = True,

        # GNN block parameters
        hidden_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        concat_heads: bool = True,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.concat_heads = concat_heads

        self.embedding = GeoEmbedding(
            num_radial=n_rbf_bond,
            num_angular=n_rbf_angle,
            hidden_channels=emb_hidden,
            out_channels=emb_dim,
            num_layers=emb_layers,
            act=emb_act,
            bias=emb_bias,
        )

        _in_channels = emb_dim
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = GeoConvBlock(
                in_channels=_in_channels,
                out_channels=hidden_channels,
                edge_attr_dim=n_rbf_angle,
                heads=heads,
                dropout=dropout,
                concat_heads=concat_heads,
            )
            self.blocks.append(block)
            _in_channels = block.out_channels

        self.out_head = nn.Linear(_in_channels, num_classes, bias=False)

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
        for block in self.blocks:
            h_bond, h_angle = block(h_bond, edge_index, edge_attr=h_angle)

        # Pooling from bonds to atoms
        bond_source = data.bond_source
        bond_target = data.bond_target
        num_atoms = data.num_atoms

        atom_index = torch.cat([bond_source, bond_target], dim=0)
        bond_embeddings = torch.cat([h_bond, h_bond], dim=0)

        h_atom = scatter(bond_embeddings, atom_index, dim=0, dim_size=num_atoms, reduce="mean")

        # Final MLP for node classification
        out = self.out_head(h_atom)

        return out
