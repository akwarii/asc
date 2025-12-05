from torch_geometric.nn import MessagePassing


class CEGANNv2Conv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_attr_dim: int,
    ) -> None:
        super().__init__(aggr="add")
