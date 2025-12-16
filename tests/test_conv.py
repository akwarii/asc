import torch
from src.models.layers.geo_conv import EdgeGatedGATv2Conv


def test_edge_gated_gatv2_conv():
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 3], [1, 0, 1, 1, 2]])
    edge_attr = torch.randn(edge_index.size(1), 7)

    conv = EdgeGatedGATv2Conv(
        in_node_channels=x.size(-1),
        in_edge_channels=edge_attr.size(-1),
        hidden_channels=16,
        out_node_channels=32,
        out_edge_channels=8,
        heads=2,
        dropout=0.1,
        norm="layernorm",
        concat=True,
    )
    out = conv(x, edge_index, edge_attr)
    assert out[0].size() == (x.size(0), 32)
    assert out[1].size() == (edge_index.size(1), 8)
