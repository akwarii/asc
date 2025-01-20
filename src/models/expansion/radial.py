import math
from typing import Any

import torch

from .envelope import ExponentialEnvelope, PolynomialEnvelope


class GaussianBasis(torch.nn.Module):
    """Reimplementation of the gaussian smearing of `torch_geometric.nn.schnet.GaussianSmearing`.

    Args:
        start: The starting value of the smearing offset.
        stop: The stopping value of the smearing offset.
        num_radial: The number of Gaussian functions to use for smearing.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_radial: int = 50,
        bond: bool = True,
    ) -> None:
        super().__init__()

        self.num_radial = num_radial
        self.bond = bond
        offset = torch.linspace(start, stop, num_radial)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist_scaled: torch.Tensor) -> torch.Tensor:  # DB
        """Forward pass of the Gaussian smearing module.

        Args:
            dist_scaled: The input scaled distance tensor.
            bond: Whether the input is a bond distance or an angle.

        Returns:
            torch.Tensor: The smearing output tensor.
        """
        if self.bond:
            dist_scaled = dist_scaled.view(-1, 1) - self.offset.view(1, -1)
        else:
            dist_scaled = dist_scaled.unsqueeze(-1).repeat(
                1, 1, self.offset.size()[0]
            ) - self.offset.view(1, -1).unsqueeze(1).repeat(1, dist_scaled.size()[-1], 1)
        return torch.exp(self.coeff * dist_scaled**2)


class RadialBesselBasis(torch.nn.Module):
    """Radial Bessel basis, as proposed in Gasteiger et al (2022). Directional Message Passing for
    Molecular Graphs (arXiv:2003.03123).

    Args:
        num_radial: The number of radial basis functions.
        stop: The cutoff value for scaling the distance.
    """

    def __init__(
        self,
        num_radial: int = 16,
        stop: float = 5.0,
    ) -> None:
        super().__init__()

        self.num_radial = num_radial

        # divide by stop ** 2 to counteract the scaling of the distances
        self.norm_factor = math.sqrt(2 / stop**3)

        self.freq = torch.nn.Parameter(
            data=torch.Tensor(math.pi * torch.arange(1, num_radial + 1) / stop),
            requires_grad=True,
        )

    def forward(self, dist_scaled: torch.Tensor) -> torch.Tensor:
        """Forward pass of the radial Bessel basis.

        Args:
            dist_scaled: The input scaled distance tensor.

        Returns:
            torch.Tensor: The radial Bessel basis output tensor.
        """
        dist_scaled = dist_scaled.view(-1, 1)
        return self.norm_factor * torch.sin(self.freq * dist_scaled) / dist_scaled


RADIAL_FUNCTIONS = {
    "gaussian": GaussianBasis,
    "bessel": RadialBesselBasis,
}
ENVELOPE_FUNCTIONS = {
    "exponential": ExponentialEnvelope,
    "polynomial": PolynomialEnvelope,
}


class RadialBasisExpansion(torch.nn.Module):
    """Radial basis expansion module. This module can be used either as the radial part of the
    3-body basis expansion or as the 2-body basis expansion.

    Args:
        num_radial: The number of radial basis functions.
        cutoff: The cutoff value for scaling the distance.
        expansion: The type of expansion function to use. Defaults to "gaussian".
        envelope: The type of envelope function to use. Defaults to None.
        expansion_kwargs: Additional keyword arguments for the expansion function.
        envelope_kwargs: Additional keyword arguments for the envelope function.
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

        self.num_radial = num_radial

        self.icutoff = 1 / cutoff

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
            self.expansion.cutoff = cutoff

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
