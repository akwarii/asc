from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import CachedLoader, NeighborLoader
from torch_geometric.nn import Linear
from torch_geometric.utils import trim_to_layer
from tqdm import tqdm

from src.models.layers.embedding import GeometricEmbedding
from src.models.layers.geo_conv import GeometricConv
from src.models.layers.readout import BondToAtomReadout


# TODO we can probably trim down some arguments
# such as hidden channels in conv and embedding output channels
class CEGANNv2(nn.Module):
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

        # Encode distances and angles
        x, edge_attr = self.embedding(x, edge_attr)

        # Convolution blocks on the line graph
        for i, conv in enumerate(self.convs):
            # Trim to sampled nodes/edges if neighbor sampling is used
            if num_sampled_nodes_per_hop is not None and num_sampled_edges_per_hop is not None:
                x, edge_index, edge_attr = trim_to_layer(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_attr,
                )

            x, edge_attr = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # During batching, num_atoms can be a tensor
        # Keep as tensor to avoid graph breaks in torch.compile
        if isinstance(num_atoms, Tensor):
            num_atoms = num_atoms.sum()

        # Pooling from bonds to atoms
        # FIXME will break when using neighbor sampling on the line graph because
        # we can't ensure we have the full bond-to-atom incidence info
        h_atom = self.readout(x, num_atoms, bond_source=bond_source)

        # Final MLP for node classification
        out = self.out_head(h_atom)

        return out

    @torch.no_grad()
    def inference_per_layer(
        self,
        layer: int,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_size: int,
    ) -> Tensor:
        """Performs inference for a single layer."""
        if layer == 0:
            x, edge_attr = self.embedding(x, edge_attr)
        else:
            # Re-embed edges as we don't propagate edge updates in inference
            edge_attr_sbf = self.embedding.sbf(edge_attr)
            edge_attr = self.embedding.edge_embedding(edge_attr_sbf)

        # TODO update signature of conv layers
        x, _ = self.convs[layer](x, edge_index, edge_attr)
        x = x[:batch_size]

        if layer == self.num_layers - 1:
            return x

        return x

    @torch.no_grad()
    def inference(
        self,
        loader: NeighborLoader,
        embedding_device: str | torch.device | None = "cpu",
        *,
        cache: bool = False,
        progress_bar: bool = True,
    ) -> Tensor:
        r"""Performs layer-wise inference on large-graphs using a
        :class:`~torch_geometric.loader.NeighborLoader`, where
        :class:`~torch_geometric.loader.NeighborLoader` should sample the
        full neighborhood for only one layer.
        This method, described in e.g., `DGI: An Easy and Efficient Framework
        for GNN Model Evaluation`, P. Yin et al., (2023), is an efficient way
        to compute the output embeddings for all nodes in the graph.

        Args:
            loader (torch_geometric.loader.NeighborLoader): A neighbor loader
                object that generates full 1-hop subgraphs, *i.e.*,
                :obj:`loader.num_neighbors = [-1]`.
            embedding_device (torch.device, optional): The device to store
                intermediate embeddings on. If intermediate embeddings fit on
                GPU, this option helps to avoid unnecessary device transfers.
                (default: :obj:`"cpu"`)
            cache (bool, optional): If set to :obj:`True`, caches intermediate
                sampler outputs for usage in later epochs.
                This will avoid repeated sampling to accelerate inference.
                (default: :obj:`False`)
            progress_bar (bool, optional): If set to :obj:`True`, displays a
                progress bar during inference. (default: :obj:`True`)
        """
        assert isinstance(loader, NeighborLoader)
        assert len(loader.dataset) == loader.data.num_nodes
        assert len(loader.node_sampler.num_neighbors) == 1
        assert not self.training

        if progress_bar:
            pbar = tqdm(total=len(loader.dataset) * len(self.convs))
            pbar.set_description("Evaluating")

        x_all = loader.data.x.to(self.device)
        if cache:

            def transform(data: Data) -> Data:
                kwargs = dict(n_id=data.n_id, batch_size=data.batch_size)
                if hasattr(data, "adj_t"):
                    kwargs["adj_t"] = data.adj_t
                else:
                    kwargs["edge_index"] = data.edge_index
                return Data.from_dict(kwargs).to(self.device)

            loader = CachedLoader(loader, device=self.device, transform=transform)  # type: ignore[assignment]

        for i in range(self.num_layers):
            xs = torch.empty(
                x_all.size(0),
                self.convs[i].node_out_channels,  # type: ignore[attr-defined]
                device=embedding_device,
                pin_memory=(embedding_device == "cpu"),
            )

            for batch in loader:
                batch_size = batch.batch_size

                n_id = batch.n_id.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_attr = batch.edge_attr.to(self.device)

                x = x_all[n_id]
                # TODO consider that both node and edge features can be updated
                x = self.inference_per_layer(i, x, edge_index, edge_attr, batch_size)

                global_id = batch.n_id.narrow(0, 0, batch_size)
                xs[global_id] = x.to(embedding_device, non_blocking=True)

                if progress_bar:
                    pbar.update(batch_size)  # type: ignore[call-arg]

            if embedding_device == "cpu":
                torch.cuda.synchronize()
            x_all = xs.to(self.device)

        # Readout from bonds to atoms
        # We assume the loader.dataset or the full graph has the necessary info
        # to map the final bond embeddings x_all back to atoms.
        # This requires bond_source corresponding to x_all.
        if hasattr(loader.data, "bond_source"):
            bond_source = loader.data.bond_source.to(self.device)
            num_atoms = loader.data.num_atoms
            x_all = self.readout.forward(x_all, num_atoms, bond_source=bond_source)
        else:
            # Fallback or error if bond_source is missing, strictly needed for CEGANN
            raise RuntimeError("bond_source missing in loader.data during inference")

        x_all = self.out_head(x_all)

        if progress_bar:
            pbar.close()  # type: ignore[call-arg]

        return x_all
