import inspect
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.resolver import activation_resolver, normalization_resolver
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax


class EdgeGatedGATv2Conv(MessagePassing):
    """Edge update (GLU gate) -> Node update (GATv2-like attention conditioned on updated edges).
    Returns updated (x, edge_attr) so BOTH evolve across layers.

    Args:
        in_node_channels (int): Number of input node channels.
        in_edge_channels (int): Number of input edge channels.
        hidden_channels (int): Hidden dimension size.
        out_node_channels (int): Number of output node channels.
        out_edge_channels (int): Number of output edge channels.
        heads (int, optional): Number of attention heads. (default: :obj:`1`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
        concat (bool, optional): Whether to concatenate multi-head outputs
            (True) or average them (False). (default: :obj:`True`)
        residual (bool, optional): Whether to use residual connections.
            (default: :obj:`True`)
        norm (str or Callable, optional): Normalization layer to use.
            (default: :obj:`"layernorm"`)
        norm_kwargs (dict, optional): Additional arguments for the normalization
            layer. (default: :obj:`None`)
        act (str or Callable, optional): Activation function to use.
            (default: :obj:`"silu"`)
        act_kwargs (dict, optional): Additional arguments for the activation
            function. (default: :obj:`None`)
    """

    def __init__(
        self,
        in_node_channels: int,
        in_edge_channels: int,
        hidden_channels: int,
        out_node_channels: int,
        out_edge_channels: int,
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

        self.x_dim = in_node_channels
        self.e_dim = in_edge_channels
        self.hidden_channels = hidden_channels
        self.out_node_channels = out_node_channels
        self.out_edge_channels = out_edge_channels
        self.heads = heads
        self.dropout = dropout
        self.concat = concat
        self.residual = residual

        # Activations
        self.act = activation_resolver(act, **(act_kwargs or {}))

        # Normalizations
        norm_layer = normalization_resolver(norm, **(norm_kwargs or {}))
        if norm_layer is None:
            norm_layer = torch.nn.Identity()

        self.norm_x = deepcopy(norm_layer)
        self.norm_e_edge = deepcopy(norm_layer)
        self.norm_e_node = deepcopy(norm_layer)

        # Edge update
        # Only one bias term is needed for uv_ since we sum the outputs
        # The layers are fused for efficiency
        self.lin_uv_l = Linear(in_node_channels, 2 * hidden_channels, bias=False)
        self.lin_uv_r = Linear(in_node_channels, 2 * hidden_channels, bias=False)
        self.lin_uv_e = Linear(in_edge_channels, 2 * hidden_channels, bias=True)

        # Edge output projection
        self.edge_out = Linear(hidden_channels, out_edge_channels, bias=True)

        # Node attention (GATv2-like)
        self.lin_lr = Linear(in_node_channels, 2 * heads * hidden_channels, bias=False)
        self.lin_e = Linear(out_edge_channels, heads * hidden_channels, bias=False)

        self.att = Parameter(torch.empty(1, heads, hidden_channels))

        # Output projection depends on head combining
        total_node_out_channels = heads * hidden_channels if concat else hidden_channels
        self.lin_out = Linear(total_node_out_channels, out_node_channels, bias=True)

        # Residuals projections
        if residual:
            self.res_x = Linear(
                in_node_channels,
                total_node_out_channels,
                bias=False,
                weight_initializer="glorot",
            )
            self.res_e = Linear(
                in_edge_channels,
                out_edge_channels,
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
        lr = self.lin_lr(x_norm)
        x_l, x_r = lr.split(channels, dim=-1)

        # Start by updating the edges
        # edge_updater_type: (uv: PairTensor, x: PairTensor, edge_attr: Tensor,
        #   res_edge_attr: OptTensor)
        edge_attr_out, alpha = self.edge_updater(
            edge_index,
            uv=(uv_l, uv_r),
            x=(x_l, x_r),
            edge_attr=edge_attr_norm,
            res_edge_attr=res_edge_attr,
        )

        # Node attention using updated edges
        # propagate_type: (x: PairTensor, alpha: Tensor)
        x_out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)  # [N,H,C]

        # Combine attention heads
        if self.concat:
            x_out = x_out.reshape(-1, heads * channels)
        else:
            x_out = x_out.mean(dim=1)

        # Project heads output.
        x_out = self.lin_out(x_out)

        # Residual connection.
        if res_x is not None:
            x_out = x_out + res_x

        return x_out, edge_attr_out

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
        dim_size: int | None,
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
        edge_attr_norm = self.norm_e_edge(edge_attr_out)

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
            f"{self.__class__.__name__}(x_dim={self.x_dim}, "
            f"e_dim={self.e_dim}, hidden={self.hidden_channels}, heads={self.heads})"
        )
