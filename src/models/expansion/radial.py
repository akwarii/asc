import math
from collections.abc import Callable
from typing import Any

import torch

from .envelope import ExponentialEnvelope, PolynomialEnvelope


class GaussianBasis(torch.nn.Module):
    """Reimplementation of the gaussian smearing of `torch_geometric.nn.schnet.GaussianSmearing`.

    Args:
        start: The starting value of the smearing offset.
        stop: The stopping value of the smearing offset.
        num_radial: The number of Gaussian functions to use for smearing.
        bond: Whether the input is a bond distance or an angle.
    """

    def __init__(
        self,
        num_radial: int = 64,
        start: float = 0.0,
        stop: float = 6.0,
    ) -> None:
        super().__init__()

        self.num_radial = num_radial
        offset = torch.linspace(start, stop, num_radial)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Gaussian smearing module.

        Args:
            dist: The input scaled distance tensor.

        Returns:
            torch.Tensor: The smearing output tensor.
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)  # type: ignore
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RadialBesselBasis(torch.nn.Module):
    r"""Radial Bessel basis, as proposed in Gasteiger et al (2022). Directional Message Passing for
    Molecular Graphs (arXiv:2003.03123).

    Args:
        num_radial: The number of radial basis functions.
        stop: The cutoff value for scaling the distance.
        trainable: Whether to train the frequencies :math:`n \pi`.
    """

    def __init__(
        self,
        num_radial: int = 8,
        stop: float = 6.0,
        *,
        trainable: bool = True,
    ) -> None:
        super().__init__()

        self.num_radial = num_radial
        self.trainable = trainable
        self.r_max = stop

        self.freq = torch.nn.Parameter(torch.empty(num_radial))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(math.pi)
        self.freq.requires_grad_(self.trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate Bessel Basis for input x.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Radial Bessel basis (shape [num_edges, num_radial]).
        """
        prefactor = 2.0 / self.r_max
        x_expanded = x.unsqueeze(-1)
        numerator = torch.sin(self.freq * x_expanded / self.r_max)
        return prefactor * numerator / x_expanded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_radial={self.num_radial}, stop={self.r_max})"


RADIAL_FUNCTIONS: dict[str, Callable] = {
    "gaussian": GaussianBasis,
    "bessel": RadialBesselBasis,
}
ENVELOPE_FUNCTIONS: dict[str, Callable] = {
    "exponential": ExponentialEnvelope,
    "polynomial": PolynomialEnvelope,
}


class RadialBasisExpansion(torch.nn.Module):
    """Radial basis expansion module. This module can be used either as the radial part of the
    3-body basis expansion or as the 2-body basis expansion.

    Args:
        num_radial: The number of radial basis functions.
        stop: The cutoff value for scaling the distance.
        expansion: The type of expansion function to use. Defaults to "gaussian".
        envelope: The type of envelope function to use. Defaults to None.
        expansion_kwargs: Additional keyword arguments for the expansion function.
        envelope_kwargs: Additional keyword arguments for the envelope function.
    """

    def __init__(
        self,
        num_radial: int = 50,
        stop: float = 5.0,
        expansion: str = "gaussian",
        envelope: str | None = None,
        expansion_kwargs: dict[str, Any] | None = None,
        envelope_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.num_radial = num_radial

        self.icutoff = 1 / stop

        if envelope_kwargs is None:
            envelope_kwargs = {}

        if envelope is None:
            self.envelope = None
        elif envelope in ENVELOPE_FUNCTIONS:
            self.envelope = ENVELOPE_FUNCTIONS[envelope](**envelope_kwargs)
        else:
            raise ValueError(
                f"Unknown envelope function '{envelope}'. "
                f"Available options are {ENVELOPE_FUNCTIONS.keys()} or None."
            )

        if expansion_kwargs is None:
            expansion_kwargs = {}
        if expansion not in RADIAL_FUNCTIONS:
            raise ValueError(
                f"Unknown expansion function '{expansion}'. "
                f"Available options are {RADIAL_FUNCTIONS.keys()}."
            )

        self.expansion = RADIAL_FUNCTIONS[expansion](
            num_radial=num_radial,
            **expansion_kwargs,
        )

        if hasattr(self.expansion, "cutoff"):
            self.expansion.cutoff = stop

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Forward pass of the radial basis expansion module.

        Args:
            dist: The input distance tensor.

        Returns:
            torch.Tensor: The output tensor after applying the radial basis expansion.
                The output tensor is of shape (len(dist), num_radial).
        """
        d_scaled = dist * self.icutoff

        if self.envelope is None:
            return self.expansion(d_scaled)

        env = self.envelope(d_scaled)
        return self.expansion(d_scaled) * env[:, None]
