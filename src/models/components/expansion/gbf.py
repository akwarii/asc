import torch
from torch import nn


class GaussianBasisExpansion(nn.Module):
    def __init__(self, gbf):
        """
        Initializes the GBFExpansion module.

        Args:
            gbf (dict): A dictionary containing the parameters for Gaussian basis function expansion.
                - dmin (float): The minimum value for the Gaussian basis function.
                - dmax (float): The maximum value for the Gaussian basis function.
                - steps (int): The number of steps for the Gaussian basis function.

        """
        super().__init__()

        self.min = gbf["dmin"]
        self.max = gbf["dmax"]
        self.steps = gbf["steps"]
        self.gamma = (self.max - self.min) / self.steps
        self.register_buffer("filters", torch.linspace(self.min, self.max, self.steps))

    def forward(self, data: torch.Tensor, bond=True) -> torch.Tensor:
        """
        Performs the forward pass of the GBFExpansion module.

        Args:
            data (torch.Tensor): The input data tensor.
            bond (bool, optional): Whether to include bond dimension. Defaults to True.

        Returns:
            torch.Tensor: The output tensor after applying the Gaussian basis function expansion.

        """
        dim = 2
        if bond:
            dim += 1
        return torch.exp(-((data.unsqueeze(dim) - self.filters) ** 2) / self.gamma**2)
