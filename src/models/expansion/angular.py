import math
from typing import Any

import torch
from torch import Tensor, nn

from .radial import GaussianBasis, RadialBasisExpansion


class RealSphHarmBasis(torch.nn.Module):
    """Real spherical harmonics basis expansion module. We only use one angle in the spherical
    coordinates, so we can set m=0 for all the spherical harmonics.

    Args:
        num_spherical: The number of spherical harmonics to use.
    """

    def __init__(self, num_spherical: int = 6) -> None:
        super().__init__()
        self.num_spherical = num_spherical

    def forward(self, theta: Tensor) -> Tensor:
        """Forward pass of the real spherical harmonics basis expansion module.

        Args:
            theta: Angle tensor in radians.

        Returns:
            Tensor: The spherical harmonics basis expansion Y_l^0.
                The output tensor is of shape `(len(theta), num_spherical)`.
        """
        # For m=0, Y_l^0(theta, phi) = sqrt((2l+1)/(4pi)) * P_l(cos(theta))
        # where P_l are Legendre polynomials.
        cos_theta = theta.cos()

        p_l = [torch.ones_like(cos_theta)]  # P_0 = 1

        if self.num_spherical > 1:
            p_l.append(cos_theta)  # P_1 = x

        for i in range(1, self.num_spherical - 1):
        # and P_{l+1}(x) = [(2l+1) x P_l(x) - l P_{l-1}(x)] / (l+1)
            p_next = ((2 * i + 1) * cos_theta * p_l[-1] - i * p_l[-2]) / (
                i + 1
            )
            p_l.append(p_next)

        res = []
        for i in range(self.num_spherical):
            prefactor = math.sqrt((2 * i + 1) / (4 * math.pi))
            res.append(prefactor * p_l[i])

        return torch.stack(res, dim=-1)


class SineBasis(nn.Module):
    """Angular basis using simple sinusoidal functions.

    Args:
        num_spherical: The number of sine basis functions to use.
    """

    def __init__(
        self,
        num_spherical: int = 32,
    ) -> None:
        super().__init__()

        self.num_spherical = num_spherical

        freqs = torch.arange(1.0, self.num_spherical + 1.0).float()
        self.register_buffer("freqs", freqs)

    def forward(self, theta: Tensor) -> Tensor:
        """Evaluate the sine basis for angles theta.

        Args:
            theta: Angle tensor in radians.

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

    def forward(self, dist: Tensor, phi: Tensor, dist_index: Tensor | None = None) -> Tensor:
        """Forward pass of the angular basis expansion module.

        Args:
            dist: The input distance tensor.
            phi: The input angle tensor in radians. The values should be in the range [0, pi].
            dist_index: Optional index tensor to map distances to angles (e.g. for triplets).

        Returns:
            The output tensor after applying the angular basis expansion. its shape is
            `(num_triplets, num_spherical, num_radial)`.
        """
        if dist_index is not None:
            dist = dist[dist_index]

        if dist.shape[0] != phi.shape[0]:
            raise ValueError(
                "dist and phi must have the same length after indexing. "
                f"Got {dist.shape[0]} and {phi.shape[0]}."
            )

        rbf = self.radial_basis(dist)  # (num_triplets, num_radial)
        abf = self.angular_basis(phi)  # (num_triplets, num_spherical)

        return rbf[:, None, :] * abf[:, :, None]  # (num_triplets, num_spherical, num_radial)
