import torch
from src.models.layers.geo_conv import GeometricConv
from tests.utils import MockGraphSpec, make_mock_pyg_graph


def test_edge_gated_gatv2_conv() -> None:
    specs = MockGraphSpec(
        num_nodes=500*14,
        num_edges=500*14*13,
        node_feat_dim=16,
        edge_feat_dim=8,
        directed=False,
        allow_self_loops=False,
        allow_duplicate_edges=False,
        device="cuda",
    )
    data = make_mock_pyg_graph(specs)

    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    assert x is not None
    assert edge_index is not None
    assert edge_attr is not None

    conv = GeometricConv(
        node_in_channels=x.size(-1),
        edge_in_channels=edge_attr.size(-1),
        hidden_channels=16,
        node_out_channels=32,
        edge_out_channels=8,
        heads=2,
        dropout=0.1,
        norm="layernorm",
        concat=True,
    )
    conv = conv.to(specs.device)
    conv = torch.compile(conv, fullgraph=True)

    out = conv(x, edge_index, edge_attr)
    assert out[0].size() == (x.size(0), 32)
    assert out[1].size() == (edge_index.size(1), 8)


if __name__ == "__main__":
    test_edge_gated_gatv2_conv()
