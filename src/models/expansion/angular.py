import math
from typing import Any

import torch
from scipy.special import sph_harm
from torch import Tensor, nn

from .radial import GaussianBasis, RadialBasisExpansion


class RealSphHarmBasis(torch.nn.Module):
    """Real spherical harmonics basis expansion module. We only use one angle in the spherical
    coordinates, so we can set m=0 for all the spherical harmonics. This implementation is based on
    the `scipy.special.sph_harm` function.

    Args:
        num_spherical: The number of spherical harmonics to use.
    """

    def __init__(self, num_spherical: int = 6) -> None:
        super().__init__()
        self.num_spherical = num_spherical

    def forward(self, phi: Tensor) -> Tensor:
        """Forward pass of the real spherical harmonics basis expansion module.

        Args:
            phi: The angle tensor in radians. The values should be in the range [0, pi].

        Returns:
            Tensor: The spherical harmonics basis expansion.
                The output tensor is of shape `(len(phi), num_spherical)`.
        """
        # TODO retrieve PyG implementation of Dimenet embedding for faster computation
        l_values = torch.arange(self.num_spherical)
        sph_harm_values = sph_harm(0, l_values[:, None], 0, phi).real
        return sph_harm_values.T


class SineBasis(nn.Module):
    """Angular basis using simple sinusoidal functions.

    Args:
        num_basis: The number of sine basis functions to use.
    """

    def __init__(
        self,
        num_basis: int = 32,
    ) -> None:
        super().__init__()

        self.num_basis = num_basis
        self.r_min = 0.0
        self.r_max = math.pi

        freqs = torch.arange(1.0, self.num_basis + 1.0).float()
        self.register_buffer("freqs", freqs)

    def forward(self, theta: Tensor) -> Tensor:
        """Evaluate the sine basis for angles theta.

        Args:
            theta: Input tensor in radian.

        Returns:
            Tensor: Sine basis.
        """
        z = theta.unsqueeze(-1) * self.freqs.view(1, -1)  # type: ignore
        return torch.sin(z)


ANGULAR_FUNCTIONS = {
    "sph_harm": RealSphHarmBasis,
    "gaussian": GaussianBasis,
    "sine": SineBasis,
}


class AngularBasisExpansion(torch.nn.Module):
    """Expansion module that combines a radial basis expansion with an angular basis expansion.

    Attributes:
        radial_basis: The radial basis expansion module.
        angular_basis: The angular basis expansion module
    """

    def __init__(
        self,
        radial_basis: RadialBasisExpansion,
        num_spherical: int = 6,
        expansion: str = "sph_harm",
        expansion_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        if expansion in ["spherical_harmonic", "spherical_harmonics"]:
            expansion = "sph_harm"
        if expansion not in ANGULAR_FUNCTIONS:
            raise ValueError(f"Unknown expansion function '{expansion}'.")

        if expansion_kwargs is None:
            expansion_kwargs = {}

        self.radial_basis = radial_basis
        self.angular_basis = ANGULAR_FUNCTIONS[expansion](
            num_spherical=num_spherical, **expansion_kwargs
        )

    # TODO handle the case where len(dist) != len(phi)
    def forward(self, dist: Tensor, phi: Tensor) -> Tensor:
        """Forward pass of the angular basis expansion module.

        Args:
            dist: The input distance tensor.
            phi: The input angle tensor in radians. The values should be in the range [0, pi].

        Returns:
            The output tensor after applying the angular basis expansion. its shape is
            `(len(dist), num_spherical, num_radial)`.
        """
        if dist.shape[0] != phi.shape[0]:
            raise ValueError(
                f"dist and phi must have the same length. Got {dist.shape[0]} and {phi.shape[0]}."
            )

        rbf = self.radial_basis(dist)  # (num_edges, num_radial)
        abf = self.angular_basis(phi)  # (num_triplets, num_spherical)

        return rbf[:, None, :] * abf[:, :, None]  # (num_triplets, num_spherical, num_radial)
