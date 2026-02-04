from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax

from src.utils.builder import normalization_builder


class GeometricConv(MessagePassing):
    """Edge update (GLU gate) -> Node update (GATv2-like attention conditioned on updated edges).
    Returns updated (x, edge_attr) so BOTH evolve across layers.

    Args:
        in_node_channels (int): Number of input node channels.
        in_edge_channels (int): Number of input edge channels.
        hidden_channels (int): Hidden dimension size.
        out_node_channels (int): Number of output node channels.
        out_edge_channels (int): Number of output edge channels.
        heads (int, optional): Number of attention heads. Default is 1.
        dropout (float, optional): Dropout probability. Default is 0.0.
        concat (bool, optional): Whether to concatenate multi-head outputs
            (True) or average them (False). Default is True.
        residual (bool, optional): Whether to use residual connections. Default is True.
        norm (str or Callable, optional): Normalization layer to use. Default is "layernorm".
        norm_kwargs (dict, optional): Additional arguments for the normalization
            layer. Default is None.
        act (str or Callable, optional): Activation function to use. Default is "silu".
        act_kwargs (dict, optional): Additional arguments for the activation
            function. Default is None.
    """

    def __init__(
        self,
        node_in_channels: int,
        edge_in_channels: int,
        hidden_channels: int,
        node_out_channels: int,
        edge_out_channels: int,
        *,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
        residual: bool = True,
        norm: str | Callable | None = "layernorm",
        norm_kwargs: dict[str, Any] | None = None,
        act: str | Callable | None = "silu",
        act_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(aggr="sum", node_dim=0, **kwargs)

        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.hidden_channels = hidden_channels
        self.node_out_channels = node_out_channels
        self.edge_out_channels = edge_out_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        self.residual = residual

        # Activations
        self.act = activation_resolver(act, **(act_kwargs or {}))

        # Normalizations
        self.norm_x = normalization_builder(norm, node_in_channels, norm_kwargs)
        self.norm_e_edge = normalization_builder(norm, edge_in_channels, norm_kwargs)
        self.norm_e_node = normalization_builder(norm, edge_out_channels, norm_kwargs)

        # Edge update (1 headed)
        # Only one bias term is needed for uv_ since we sum the outputs
        # The layers are fused for efficiency
        fused_channels = 2 * hidden_channels
        self.lin_uv_l = Linear(node_in_channels, fused_channels, bias=False)
        self.lin_uv_r = Linear(node_in_channels, fused_channels, bias=False)
        self.lin_uv_e = Linear(edge_in_channels, fused_channels, bias=True)

        # Edge output projection
        self.edge_out = Linear(hidden_channels, edge_out_channels, bias=True)

        # Node attention (GATv2-like, multi-headed)
        self.lin_lr = Linear(node_in_channels, heads * fused_channels, bias=False)
        self.lin_e = Linear(edge_out_channels, heads * hidden_channels, bias=False)
        self.att = Parameter(torch.empty(1, heads, hidden_channels))

        # Output projection depends on head combining
        total_node_out_channels = hidden_channels * (heads if concat else 1)
        self.lin_out = Linear(total_node_out_channels, node_out_channels, bias=True)

        # Residuals projections
        if self.residual:
            self.res_x = Linear(
                node_in_channels,
                node_out_channels,
                bias=False,
                weight_initializer="glorot",
            )
            self.res_e = Linear(
                edge_in_channels,
                edge_out_channels,
                bias=False,
                weight_initializer="glorot",
            )
        else:
            self.register_parameter("res_x", None)
            self.register_parameter("res_e", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        self.lin_uv_l.reset_parameters()
        self.lin_uv_r.reset_parameters()
        self.lin_uv_e.reset_parameters()
        self.edge_out.reset_parameters()

        self.lin_lr.reset_parameters()
        self.lin_e.reset_parameters()

        if self.residual:
            self.res_x.reset_parameters()
            self.res_e.reset_parameters()

        glorot(self.att)

        self.lin_out.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the module."""
        heads, channels = self.heads, self.hidden_channels

        # Project residuals if needed
        res_x: OptTensor = None
        res_edge_attr: OptTensor = None
        if self.residual:
            res_x = self.res_x(x)
            res_edge_attr = self.res_e(edge_attr)

        # PreNorm for edge branch
        x_norm = self.norm_x(x)
        edge_attr_norm = self.norm_e_edge(edge_attr)

        # Fused projection for edge update
        uv_l = self.lin_uv_l(x_norm)
        uv_r = self.lin_uv_r(x_norm)

        # Fused projection for attention coefficients
        lr: Tensor = self.lin_lr(x_norm)
        x_l, x_r = lr.chunk(2, dim=-1)
        x_l = x_l.view(-1, heads, channels)
        x_r = x_r.view(-1, heads, channels)

        # Start by updating the edges
        # edge_updater_type: (uv: PairTensor, x: PairTensor, edge_attr: Tensor,
        #   res_edge_attr: OptTensor)
        out_edge_attr, alpha = self.edge_updater(
            edge_index,
            uv=(uv_l, uv_r),
            x=(x_l, x_r),
            edge_attr=edge_attr_norm,
            res_edge_attr=res_edge_attr,
        )

        # Node attention using updated edges
        # propagate_type: (x: PairTensor, alpha: Tensor)
        out_x = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        # Combine attention heads
        if self.concat:
            out_x = out_x.view(-1, self.heads * self.hidden_channels)
        else:
            out_x = out_x.mean(dim=1)

        # Project heads output
        out_x = self.lin_out(out_x)

        # Residual connection
        if res_x is not None:
            out_x = out_x + res_x

        return out_x, out_edge_attr

    def edge_update(  # type: ignore
        self,
        x_j: Tensor,
        x_i: Tensor,
        uv_j: Tensor,
        uv_i: Tensor,
        edge_attr: Tensor,
        res_edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        dim_size: int | None,  # PyG doesn't handle python 3.10+ union types signatures yet
    ) -> tuple[Tensor, Tensor]:
        """Edge update with GLU-like gating mechanism."""
        # Fused projection to compute the gate inputs
        gate_inputs: Tensor = uv_i + uv_j + self.lin_uv_e(edge_attr)
        u, v = gate_inputs.chunk(2, dim=-1)

        # Gated edge update
        edge_attr_out = self.edge_out(self.act(u) * torch.sigmoid(v))

        # Dropout on edge features
        if self.dropout > 0.0 and self.training:
            edge_attr_out = F.dropout(edge_attr_out, p=self.dropout, training=self.training)

        # Residual connection
        if res_edge_attr is not None:
            edge_attr_out = edge_attr_out + res_edge_attr

        # PreNorm for attention computation
        edge_attr_norm = self.norm_e_node(edge_attr_out)

        # Initial transform of updated edge features
        if edge_attr_norm.dim() == 1:
            edge_attr_norm = edge_attr_norm.view(-1, 1)
        edge_attr_norm = self.lin_e(edge_attr_norm)
        edge_attr_norm = edge_attr_norm.view(-1, self.heads, self.hidden_channels)

        # This is equivalent (but more efficient) to W(x_i || x_j || e_ij)
        x = x_i + x_j + edge_attr_norm

        # Contrary to GAT(v2), we allow for an activation different than LeakyReLU
        logits = (self.act(x) * self.att).sum(dim=-1)
        alpha = softmax(logits, index, ptr, dim_size)  # normalize over incoming edges

        # Dropout on attention scores
        if self.dropout > 0.0 and self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return edge_attr_out, alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:  # type: ignore
        """Computes messages from node j to node i."""
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_node_channels={self.node_in_channels}, "
            f"in_edge_channels={self.edge_in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"out_node_channels={self.node_out_channels}, "
            f"out_edge_channels={self.edge_out_channels}, "
            f"heads={self.heads})"
        )
