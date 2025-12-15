from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class GeoConv(MessagePassing):
    r"""Multi-head GATv2-like convolution on a line-graph where:
      - nodes = distances (bonds),
      - edges = angles between bonds.

    Inputs:
        x:          [N, F_d_in]       node features (distances)
        edge_index: [2, E]            connectivity (angles)
        edge_attr:  [E, F_theta_in]   edge features (angles)

    Features:
      - Multi-head GATv2 attention on triplets z_e = [theta_e || x_src || x_dst].
      - Updates both nodes (distances) and edges (angles).
      - Pre-norm (LayerNorm on x, edge_attr before attention).
      - Residual connections with optional linear projection if dims differ.
      - Dropout on attention and feature updates.
      - User-defined activations for attention and output.

    Args:
        node_dim:   F_d_in
        edge_dim:   F_theta_in
        out_node_channels:  F_d_out (total, after combining heads)
        out_edge_channels:  F_theta_out (total, after combining heads)
        att_hidden_channels: size of hidden dim in attention MLP per head
        heads:              number of attention heads
        concat_heads:       if True, concatenate heads (default); else average
        residual:           enable residual connections
        dropout:            dropout probability on attention coefficients
        att_activation:     activation for attention MLP (default: LeakyReLU)
        out_activation:     activation for feature updates (default: Softplus)
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_node_channels: int,
        out_edge_channels: int,
        *,
        dropout: float = 0.0,
        att_act: Callable | str = "leaky_relu",
        att_act_kwargs: dict[str, Any] | None = None,
        out_act: Callable | str = "softplus",
        out_act_kwargs: dict[str, Any] | None = None,
        residual: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__(aggr="sum", node_dim=0)

        self.in_channels = node_dim
        self.edge_dim = edge_dim
        self.out_node_channels = out_node_channels
        self.out_edge_channels = out_edge_channels
        self.residual = residual
        self.dropout = dropout

        # Size of concatenated triplet feature: [theta || x_src || x_dst]
        triplet_in = edge_dim + 2 * node_dim

        self.lin_att = Linear(triplet_in, out_node_channels, bias=True)
        # Attention vectors per head: [heads, att_hidden]
        self.att = Parameter(torch.empty(1, out_node_channels))

        # Feature transforms for messages (per head, shared across heads via vectorization)
        self.lin_edge = Linear(edge_dim, out_edge_channels, bias=True)
        self.lin_node = Linear(out_node_channels, out_node_channels, bias=True)

        # Residual projections if needed
        if residual and node_dim != out_node_channels:
            self.lin_node_res = Linear(node_dim, out_node_channels, bias=False)
        else:
            self.lin_node_res = None

        if residual and edge_dim != out_edge_channels:
            self.lin_edge_res = Linear(edge_dim, out_edge_channels, bias=False)
        else:
            self.lin_edge_res = None

        # Activations
        self.att_act = activation_resolver(att_act, **(att_act_kwargs or {}))
        self.out_act = activation_resolver(out_act, **(out_act_kwargs or {}))

        if bias:
            self.bias = Parameter(torch.empty(out_node_channels + out_edge_channels))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        super().reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin_att.weight)
        torch.nn.init.zeros_(self.lin_att.bias)
        torch.nn.init.xavier_uniform_(self.lin_edge.weight)
        torch.nn.init.zeros_(self.lin_edge.bias)
        torch.nn.init.xavier_uniform_(self.lin_node.weight)
        torch.nn.init.zeros_(self.lin_node.bias)
        torch.nn.init.xavier_uniform_(self.att)

        if self.lin_node_res is not None:
            torch.nn.init.xavier_uniform_(self.lin_node_res.weight)
        if self.lin_edge_res is not None:
            torch.nn.init.xavier_uniform_(self.lin_edge_res.weight)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the module.

        Args:
        x:          [N, F_d_in]
        edge_index: [2, E]
        edge_attr:  [E, F_theta_in]
        """
        row, col = edge_index  # row: src, col: dst

        # ----- Residual bases -----
        x_residual = x
        edge_residual = edge_attr

        num_edges = edge_attr.size(0)

        # ----- Triplet features z_e = [theta_e || x_src || x_dst] -----
        z = torch.cat([edge_attr, x[row], x[col]], dim=-1)  # [E, F_theta + 2*F_d]

        # ----- GATv2 attention MLP -----
        # lin_att: [E, triplet_in] -> [E, heads * att_hidden]
        u = self.lin_att(z)  # [E, heads * H]
        u = self.att_act(u)  # nonlinear
        u = u.view(num_edges, self.att_hidden_channels)  # [E, H_heads, H]

        # Attention scores per head
        # att_vec: [heads, H]; u: [E, heads, H]
        scores = (u * self.att.unsqueeze(0)).sum(dim=-1)  # [E, heads]

        # Normalize over incoming edges to each destination node, per head
        alpha = softmax(scores, col)  # [E, heads]

        # Dropout on attention
        if self.dropout > 0.0:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # ----- Edge messages (angles) -----
        # u_flat: [E*heads, H]
        u_flat = u.reshape(num_edges, self.att_hidden_channels)

        # Per-head edge transform: [E*heads, H] -> [E*heads, edge_out_per_head]
        m_edge_flat = self.lin_edge(u_flat)
        m_edge = m_edge_flat.view(
            num_edges, self.edge_out_per_head
        )  # [E, heads, edge_out_per_head]

        # Weight by attention coefficients
        m_edge_weighted = alpha.unsqueeze(-1) * m_edge  # [E, heads, edge_out_per_head]

        if self.concat_heads:
            edge_msg = m_edge_weighted.reshape(
                num_edges, self.edge_out_per_head
            )  # [E, F_theta_out]
        else:
            edge_msg = m_edge_weighted.sum(dim=1)  # [E, F_theta_out]

        # Feature activation + dropout for edge updates
        edge_delta = self.out_act(edge_msg)

        # Residual connection for edges
        if self.residual:
            if self.lin_edge_res is not None:
                edge_residual_proj = self.lin_edge_res(edge_residual)
            else:
                edge_residual_proj = edge_residual
            edge_out = edge_residual_proj + edge_delta
        else:
            edge_out = edge_delta

        # ----- Node messages (distances) -----
        # Per-head node transform: [E*heads, H] -> [E*heads, node_out_per_head]
        m_node_flat = self.lin_node(u_flat)
        m_node = m_node_flat.view(
            num_edges, self.node_out_per_head
        )  # [E, heads, node_out_per_head]

        # Weight by attention
        m_node_weighted = alpha.unsqueeze(-1) * m_node  # [E, heads, node_out_per_head]

        if self.concat_heads:
            m_node_total = m_node_weighted.reshape(
                num_edges, self.node_out_per_head
            )  # [E, F_d_out]
        else:
            m_node_total = m_node_weighted.sum(dim=1)  # [E, F_d_out]

        # Propagate these precomputed messages to destination nodes
        x_msg = self.propagate(edge_index, m_node=m_node_total, size=(x.size(0), x.size(0)))
        # x_msg: [N, F_d_out]

        # Feature activation
        x_delta = self.out_act(x_msg)

        # Residual connection for nodes
        if self.residual:
            if self.lin_node_res is not None:
                x_residual_proj = self.lin_node_res(x_residual)
            else:
                x_residual_proj = x_residual
            x_out = x_residual_proj + x_delta
        else:
            x_out = x_delta

        return x_out, edge_out

    def message(self, x_j: Tensor) -> Tensor:
        """Compute messages for node updates."""
        return x_j


class GATv2Conv(MessagePassing):
    r"""The GATv2 operator from the `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ paper, which fixes the
    static attention problem of the standard
    :class:`~torch_geometric.conv.GATConv` layer.
    Since the linear layers in the standard GAT are applied right after each
    other, the ranking of attended nodes is unconditioned on the query node.
    In contrast, in :class:`GATv2`, every node can attend to any other node.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i) \cup \{ i \}}
        \alpha_{i,j}\mathbf{\Theta}_{t}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_j
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i + \mathbf{\Theta}_{t} \mathbf{x}_k
        \right)\right)}.

    If the graph has multi-dimensional edge features :math:`\mathbf{e}_{i,j}`,
    the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i
        + \mathbf{\Theta}_{t} \mathbf{x}_j
        + \mathbf{\Theta}_{e} \mathbf{e}_{i,j}
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathbf{a}^{\top}\mathrm{LeakyReLU}\left(
        \mathbf{\Theta}_{s} \mathbf{x}_i
        + \mathbf{\Theta}_{t} \mathbf{x}_k
        + \mathbf{\Theta}_{e} \mathbf{e}_{i,k}]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities in case of a bipartite graph.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default: :obj:`None`)
        fill_value (float or torch.Tensor or str, optional): The way to
            generate edge features of self-loops
            (in case :obj:`edge_dim != None`).
            If given as :obj:`float` or :class:`torch.Tensor`, edge features of
            self-loops will be directly given by :obj:`fill_value`.
            If given as :obj:`str`, edge features of self-loops are computed by
            aggregating all features of edges that point to the specific node,
            according to a reduce operation. (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"min"`, :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"mean"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge,
            *i.e.* :math:`\mathbf{\Theta}_{s} = \mathbf{\Theta}_{t}`.
            (default: :obj:`False`)
        residual (bool, optional): If set to :obj:`True`, the layer will add
            a learnable skip-connection. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})` or
          :math:`((|\mathcal{V}_t|, H * F_{out})` if bipartite.
          If :obj:`return_attention_weights=True`, then
          :math:`((|\mathcal{V}|, H * F_{out}),
          ((2, |\mathcal{E}|), (|\mathcal{E}|, H)))`
          or :math:`((|\mathcal{V_t}|, H * F_{out}), ((2, |\mathcal{E}|),
          (|\mathcal{E}|, H)))` if bipartite
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        *,
        heads: int = 1,
        concat: bool = True,
        act_att: Callable | str = "leaky_relu",
        act_att_kwargs: dict[str, Any] | None = None,
        act_out: Callable | str = "relu",
        out_act_kwargs: dict[str, Any] | None = None,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        fill_value: float | Tensor | str = "mean",
        bias: bool = True,
        residual: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual

        # _l and _r refer to source and target nodes, respectively
        self.lin_l = Linear(
            in_channels, heads * out_channels, bias=bias, weight_initializer="glorot"
        )
        self.lin_r = Linear(
            in_channels, heads * out_channels, bias=bias, weight_initializer="glorot"
        )
        self.lin_edge = Linear(
            edge_dim, heads * out_channels, bias=False, weight_initializer="glorot"
        )

        self.att = Parameter(torch.empty(1, heads, out_channels))

        # The number of output channels
        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer="glorot",
            )
        else:
            self.register_parameter("res", None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter("bias", None)

        # Activations
        self.act_att = activation_resolver(act_att, **(act_att_kwargs or {}))
        self.act_out = activation_resolver(act_out, **(out_act_kwargs or {}))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_edge.reset_parameters()
        if self.res is not None:
            self.res.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor): The edge indices.
            edge_attr (torch.Tensor): The edge features.
        """
        assert x.dim() == 2

        H, C = self.heads, self.out_channels

        res: OptTensor = None
        if self.res is not None:
            res = self.res(x)

        # Initial linear transformations, reshaped for multi-head attention
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)

        if self.add_self_loops:
            num_nodes = x_l.size(0)
            if x_r is not None:
                num_nodes = min(num_nodes, x_r.size(0))
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, fill_value=self.fill_value, num_nodes=num_nodes
            )

        # Compute attention coefficients defined in `edge_update` method
        alpha = self.edge_updater(edge_index, x=(x_l, x_r), edge_attr=edge_attr)

        # Propagate the messages defined by `message` method.
        # This function also takes care of the aggregation and the update of
        # node embeddings.
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        return out

    def edge_update(  # type: ignore
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: OptTensor,
        dim_size: int | None,
    ) -> Tensor:
        """Computes attention coefficients for each edge."""
        # Initial transform of edge features
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        edge_attr = self.lin_edge(edge_attr)
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)

        # This is equivalent to W(x_i || x_j || e_ij) since we have separate
        # linear layers for x_i, x_j, and e_ij
        x = x_i + x_j + edge_attr

        # Contrary to GAT(v2), we allow for an activation different than LeakyReLU
        x = self.act_att(x)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, dim_size) # normalize over incoming edges

        # Dropout on attention scores
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:  # type: ignore
        """Computes messages from node j to node i."""
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class GeoConvBlock(torch.nn.Module):
    """Residual block with GeoConv layer, LayerNorm, and Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_attr_dim: int,
        *,
        heads: int = 4,
        concat_heads: bool = True,
        average_last_heads: bool = True,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()

        self.conv = GeoConv(
            node_dim=in_channels,
            edge_dim=edge_attr_dim,
            out_node_channels=out_channels,
            out_edge_channels=edge_attr_dim,
            heads=heads,
            concat_heads=concat_heads,
            residual=True,
            dropout=dropout,
        )

        self.node_norm = torch.nn.LayerNorm(out_channels)
        self.edge_norm = torch.nn.LayerNorm(edge_attr_dim)
        self.dropout = torch.nn.Dropout(dropout)

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.conv.in_channels

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.conv.out_channels

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the block."""
        x_out = self.node_norm(x)
        edge_attr_out = self.edge_norm(edge_attr)

        x_out, edge_attr_out = self.conv(x_out, edge_index, edge_attr_out)

        x_out = self.dropout(x_out)
        edge_attr_out = self.dropout(edge_attr_out)

        return x_out, edge_attr_out
