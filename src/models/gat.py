from collections.abc import Callable
from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GAT, MLP

from src.models.expansion.radial import GaussianBasis


class GATClassifier(GAT):  # noqa
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        num_radial: int,
        dropout: float = 0.0,
        act: str | Callable | None = "relu",
        act_kwargs: dict[str, Any] | None = None,
        norm: str | Callable | None = None,
        norm_kwargs: dict[str, Any] | None = None,
        heads: int = 1,
        negative_slope: float = 0.2,
        share_weights: bool = False,
        residual: bool = False,
        classification_units: int | None = None,
        classification_layers: int | None = 1,
        **kwargs,
    ) -> None:
        v2 = kwargs.pop("v2", True)
        add_self_loops = kwargs.pop("add_self_loops", False)
        edge_dim = num_radial

        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            edge_dim=edge_dim,
            dropout=dropout,
            act=act,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            heads=heads,
            negative_slope=negative_slope,
            share_weights=share_weights,
            residual=residual,
            v2=v2,
            add_self_loops=add_self_loops,
            **kwargs,
        )

        self.mlp = MLP(
            in_channels=-1,
            hidden_channels=classification_units,
            num_layers=classification_layers,
            out_channels=out_channels,
            dropout=dropout,
            plain_last=True,
        )

        self.rbf = GaussianBasis(num_radial=num_radial)
        self.sbf = GaussianBasis(num_radial=num_radial)

    def forward(  # noqa
        self,
        data: Data,
        batch: torch.Tensor | None = None,
        batch_size: int | None = None,
        num_sampled_nodes_per_hop: list[int] | None = None,
        num_sampled_edges_per_hop: list[int] | None = None,
    ) -> torch.Tensor:
        assert data.x is not None
        assert data.edge_attr is not None
        assert data.edge_index is not None

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x_emb = self.rbf(x)
        edge_attr_emb = self.sbf(edge_attr)

        emb = super().forward(
            x=x_emb,
            edge_index=edge_index,
            edge_attr=edge_attr_emb,
            batch=batch,
            batch_size=batch_size,
            num_sampled_nodes_per_hop=num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop=num_sampled_edges_per_hop,
        )

        # Reduce the output to the number of atoms
        k = edge_attr.size(0) // x.size(0) + 1
        num_atoms = x.size(0) // k

        out = torch.sum(emb.view(num_atoms, k, self.out_channels), dim=1)
        return self.mlp(out)
