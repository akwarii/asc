import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter

from src.models.expansion.radial import RadialBesselBasis


class PaiNNRadial(nn.Module):
    """Rotation invariant radial filter that processes the distance between neighbors.

    Args:
        num_radial: Number of radial basis functions.
        cutoff: Cutoff distance for the radial basis functions.
        hidden_channels: Dimensionality of the hidden scalar features.
    """

    def __init__(self, num_radial: int, hidden_channels: int, cutoff: float = 6.0) -> None:
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.hidden_channels = hidden_channels

        self.rbf_filter = nn.Sequential(
            RadialBesselBasis(num_radial, cutoff),
            Linear(num_radial, 3 * hidden_channels),  # we split into 3 parts later
        )

    def forward(self, dist_mag: Tensor) -> Tensor:
        """Forward pass for the radial filter.

        Args:
            dist_mag: Distance magnitudes (shape [num_edges, 1]).
        """
        return self.rbf_filter(dist_mag)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self.rbf_filter)[1:-1]})"


class PaiNNMessage(nn.Module):
    """PaiNN message block. It processes scalar and vector features together, generating messages
    for both streams.

    Args:
        hidden_channels: Dimensionality of the hidden scalar features.
        dropout: Dropout rate for the internal MLP.
        scale_factor: Scaling factor for the message passing. To avoid exploding gradients, it is
            recommended to set this to 1 / num_neighbors.
    """

    def __init__(
        self, hidden_channels: int, dropout: float = 0.1, scale_factor: float = 1.0
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.scale_factor = scale_factor
        self.dropout = dropout

        self.phi = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            Linear(hidden_channels, 3 * hidden_channels),
        )
        self.rms_norm = nn.RMSNorm(hidden_channels)

    def forward(
        self, s: Tensor, v: Tensor, edge_index: Tensor, rbf_filter: Tensor, edge_vector: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward pass for the message block. The variable names are chosen to reflect the
        original PaiNN paper.

        Args:
            s: Scalar features (shape [num_nodes, 1, hidden_channels]).
            v: Vector features (shape [num_nodes, 3, hidden_channels]).
            edge_index: Edge indices (shape [2, num_edges]).
            rbf_filter: Radial filter (shape [num_edges, 1, 3 * hidden_channels]).
            edge_vector: Distance unit vectors (shape [num_edges, 3]).
        """
        assert (
            rbf_filter.shape[-1] == 3 * self.hidden_channels
        ), "Edge filter output dimension must be 3x hidden_channels"

        i, j = edge_index

        s_norm = self.rms_norm(s)

        # since linear and gather are commutative, we can apply the linear layer first
        phi_s = self.phi(s_norm)
        filter = phi_s[j] * rbf_filter

        # split into scalar, vector, and gate
        m_s, m_vv, m_vs = torch.chunk(filter, chunks=3, dim=-1)

        # Scalar message
        ds = scatter(m_s, i, dim=0, dim_size=s.size(0), reduce="sum")

        # Vector message
        # gate = v[j] * m_vv + m_vs * edge_vector[..., None]
        gate = torch.einsum("edc, ec -> edc", v[j], m_vv.squeeze(1))
        gate.addcmul_(edge_vector.unsqueeze(-1), m_vs)

        dv = scatter(gate, i, dim=0, dim_size=v.size(0), reduce="sum")

        # Residual
        s = s + ds * self.scale_factor
        v = v + dv * self.scale_factor

        return s, v


class PaiNNUpdate(nn.Module):
    """PaiNN update block. It takes the output from the message block and updates the scalar and
    vector features.

    Args:
        hidden_channels: Dimensionality of the hidden scalar features.
        dropout: Dropout rate for the internal MLP.
    """

    def __init__(self, hidden_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.update_net = nn.Sequential(
            Linear(2 * hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            Linear(hidden_channels, 3 * hidden_channels),
        )
        self.v_proj = Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.rms_norm = nn.RMSNorm(hidden_channels)

    def forward(self, s: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for the update block.

        Args:
            s: Scalar features (shape [num_nodes, 1, hidden_channels]).
            v: Vector features (shape [num_nodes, 3, hidden_channels]).
        """
        s_norm = self.rms_norm(s)

        u, w = torch.chunk(self.v_proj(v), chunks=2, dim=-1)
        w_norm = w.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8).sqrt()

        context = torch.cat([s_norm, w_norm], dim=-1)
        filter = self.update_net(context)

        a_ss, a_vv, a_sv = torch.chunk(filter, chunks=3, dim=-1)

        # scaling functions are used as nonlinearity
        dv = a_vv * u

        # ds = a_ss + a_sv * torch.sum(u * w, dim=1, keepdim=True)
        uw_dot = torch.einsum("ndc, ndc -> nc", u, w).unsqueeze(1)
        ds = torch.addcmul(a_ss, a_sv, uw_dot)

        # Residuals
        s = s + ds
        v = v + dv

        return s, v


class PaiNNHead(nn.Module):
    """PaiNN head block. It takes the scalar features and produces predictions.

    Args:
        hidden_channels: Dimensionality of the hidden scalar features.
        num_classes: Number of output classes.
        dropout: Dropout rate for the internal MLP.
    """

    def __init__(self, hidden_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_classes = out_channels

        self.mlp = nn.Sequential(
            Linear(hidden_channels, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            Linear(hidden_channels, out_channels),
        )

    def forward(self, s: Tensor) -> Tensor:
        """Forward pass for the head block. It only uses the scalar features for prediction.

        Args:
            s: Scalar features (shape [num_nodes, hidden_channels]).
        """
        return self.mlp(s)


class PaiNN(nn.Module):
    """PaiNN model adapted for classification tasks. It simply removes the final reduce to keep the
    atomic representation.

    Args:
        num_classes: Number of output classes.
        num_species: Number of unique atomic species (default 119 for all elements).
        num_radial: Number of radial basis functions (default 4).
        num_layers: Number of message-passing layers (default 3).
        hidden_channels: Dimensionality of the hidden scalar features (default 128).
        dropout: Dropout rate for the internal MLPs (default 0.1).
        scale_factor: Scaling factor for the message passing (default 1.0). To avoid exploding
            gradients, it is recommended to set this to 1 / num_neighbors.
    """

    def __init__(
        self,
        out_channels: int,
        num_species: int = 119,
        num_radial: int = 4,
        num_layers: int = 3,
        hidden_channels: int = 128,
        dropout: float = 0.1,
        scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.scale_factor = scale_factor

        self.embedding = nn.Embedding(num_species, hidden_channels)
        self.rbf = PaiNNRadial(num_radial, hidden_channels, cutoff=6.0)

        # Separate message and update blocks for better modularity
        self.message_blocks = nn.ModuleList(
            [PaiNNMessage(hidden_channels, dropout, scale_factor) for _ in range(num_layers)]
        )
        self.update_blocks = nn.ModuleList(
            [PaiNNUpdate(hidden_channels, dropout) for _ in range(num_layers)]
        )
        self.head = PaiNNHead(hidden_channels, out_channels, dropout)

    def forward(self, data: Data) -> Tensor:
        """Forward pass for the PaiNN model.

        Args:
            data: PyG Data object containing:
                z: Atomic numbers (shape [num_nodes]).
                edge_index: Edge indices (shape [2, num_edges]).
                edge_attr: Distance vector (shape [num_edges, 3]).
        """
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_attr is not None

        z, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # dist_mag = torch.linalg.norm(edge_attr, dim=1, keepdim=True)
        dist_mag = edge_attr.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        edge_unit_vec = edge_attr / dist_mag

        s = self.embedding(z).unsqueeze(1)
        v = torch.zeros(s.size(0), 3, s.size(2), device=s.device)

        rbf_filter = self.rbf(dist_mag)
        for message, update in zip(self.message_blocks, self.update_blocks):
            s, v = message(s, v, edge_index, rbf_filter, edge_unit_vec)
            s, v = update(s, v)

        out = self.head(s.squeeze(1))

        return out

    #TODO this is not working
    def inference(self, data: Data) -> Tensor:
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_attr is not None

        z, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        dist_mag = edge_attr.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        edge_unit_vec = edge_attr / dist_mag

        s = self.embedding(z).unsqueeze(1)
        v = torch.zeros(s.size(0), 3, s.size(2), device=s.device)

        rbf_filter_full = self.rbf(dist_mag)

        is_hierarchical = hasattr(data, "num_sampled_nodes") and data.num_sampled_nodes is not None
        for i in range(self.num_layers):
            if is_hierarchical:
                # Get counts for current layer (indexing from furthest to target)
                node_count = data.num_sampled_nodes[i]
                edge_count = data.num_sampled_edges[i]
                print(data.num_nodes, node_count, data.num_edges, edge_count)

                # Slice graph and pre-calculated features
                l_edge_index = edge_index[:, :edge_count]
                l_edge_unit_vec = edge_unit_vec[:edge_count]
                l_rbf_filter = rbf_filter_full[:edge_count]

                # Slice active features
                s_active = s[:node_count]
                v_active = v[:node_count]
            else:
                l_edge_index = edge_index
                l_edge_unit_vec = edge_unit_vec
                l_rbf_filter = rbf_filter_full
                s_active = s
                v_active = v

            s_active, v_active = self.message_blocks[i](
                s_active, v_active, l_edge_index, l_rbf_filter, l_edge_unit_vec
            )

            s_active, v_active = self.update_blocks[i](s_active, v_active)

            if is_hierarchical:
                s[:node_count] = s_active
                v[:node_count] = v_active
            else:
                s, v = s_active, v_active

        # Head only cares about the target nodes (the batch)
        out = self.head(s[: data.batch_size].squeeze(1))

        return out
