from typing import Any
import torch

from .envelope import ExponentialEnvelope, PolynomialEnvelope, DummyEnvelope


class GaussianBasis(torch.nn.Module):
    """
    Reimplementation of the gaussian smearing of `torch_geometric.nn.schnet.GaussianSmearing`.

    Args:
        start (float): The starting value of the smearing offset.
        stop (float): The stopping value of the smearing offset.
        num_radial (int): The number of Gaussian functions to use for smearing.
    """
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_radial: int = 50,
    ) -> None:
        super().__init__()

        offset = torch.linspace(start, stop, num_radial)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", torch.linspace(self.min, self.max, self.steps))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Gaussian smearing module.

        Args:
            dist (torch.Tensor): The input distance tensor.

        Returns:
            torch.Tensor: The smearing output tensor.
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))



EXPANSION_FUNCTIONS = {
    "gaussian": GaussianBasis,
}
ENVELOPE_FUNCTIONS = {
    "exponential": ExponentialEnvelope,
    "polynomial": PolynomialEnvelope,
    "dummy": DummyEnvelope,
}


class RadialBasisExpansion(torch.nn.Module):
    """
    Radial basis expansion module.

    Args:
        num_radial (int): The number of radial basis functions.
        cutoff (float): The cutoff value for scaling the distance.
        expansion (str, optional): The type of expansion function to use. Defaults to "gaussian".
        envelope (str, optional): The type of envelope function to use. Defaults to None.
        expansion_kwargs (dict[str, Any] | None, optional): Additional keyword arguments for the expansion function. Defaults to None.
        envelope_kwargs (dict[str, Any] | None, optional): Additional keyword arguments for the envelope function. Defaults to None.
    """

    def __init__(
        self,
        num_radial: int = 50,
        cutoff: float = 5.0,
        expansion: str = "gaussian",
        envelope: str | None = None,
        expansion_kwargs: dict[str, Any] | None = None,
        envelope_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.icutoff = 1 / cutoff
        
        if expansion_kwargs is None:
            expansion_kwargs = {}
        if envelope_kwargs is None:
            envelope_kwargs = {}
        
        if envelope is None:
            envelope = "dummy"
        if envelope not in ENVELOPE_FUNCTIONS:
            raise ValueError(f"Unknown envelope function '{envelope}'.")
        self.envelope = ENVELOPE_FUNCTIONS[envelope](**envelope_kwargs)
        
        if expansion not in EXPANSION_FUNCTIONS:
            raise ValueError(f"Unknown expansion function '{expansion}'.")
        self.expansion = EXPANSION_FUNCTIONS[expansion](
            num_radial=num_radial,
            cutoff=cutoff,
            **expansion_kwargs,
        )
        
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the radial basis expansion module.

        Args:
            dist (torch.Tensor): The input distance tensor.

        Returns:
            torch.Tensor: The output tensor after applying the radial basis expansion.
                The output tensor is of shape (num_edges, num_radial).
        """
        d_scaled = dist * self.icutoff
        env = self.envelope(d_scaled)
        return self.expansion(d_scaled) * env[:, None]