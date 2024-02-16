import torch
import torch.nn as nn


class GraphNorm(nn.Module):
    r"""Applies graph normalization over individual graphs as described in the
    `"GraphNorm: A Principled Approach to Accelerating Graph Neural Network
    Training" <https://arxiv.org/abs/2009.03294>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} - \alpha \odot
        \textrm{E}[\mathbf{x}]}
        {\sqrt{\textrm{Var}[\mathbf{x} - \alpha \odot \textrm{E}[\mathbf{x}]]
        + \epsilon}} \odot \gamma + \beta

    where :math:`\alpha` denotes parameters that learn how much information
    to keep in the mean.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: `1e-5`)

    Attributes:
        in_channels (int): Size of each input sample.
        eps (float): A value added to the denominator for numerical stability.
        weight (torch.Parameter): Learnable weight parameter.
        bias (torch.Parameter): Learnable bias parameter.
    """

    def __init__(self, in_channels: int, eps: float = 1e-5) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(self.in_channels))
        self.bias = nn.Parameter(torch.zeros(self.in_channels))

    def forward(self, graph: "DGLGraph", tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GraphNorm layer.

        Args:
            graph (DGLGraph): The input graph.
            tensor (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        batch_list = graph.batch_num_nodes()
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)

        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)

        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)

        return self.weight * sub / std + self.bias
