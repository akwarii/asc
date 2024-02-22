from typing import Any

import torch
from scipy.special import sph_harm

from .radial import GaussianBasis, RadialBasisExpansion


class RealSphHarmBasis(torch.nn.Module):
    """Real spherical harmonics basis expansion module. We only use one angle in the spherical
    coordinates, so we can set m=0 for all the spherical harmonics. This implementation is based on
    the `scipy.special.sph_harm` function.

    Args:
        num_spherical (int): The number of spherical harmonics to use.
    """

    def __init__(self, num_spherical: int = 6) -> None:
        super().__init__()
        self.num_spherical = num_spherical

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Forward pass of the real spherical harmonics basis expansion module.

        Args:
            phi (torch.Tensor): The angle tensor in radians.
                The values should be in the range [0, pi].

        Returns:
            torch.Tensor: The spherical harmonics basis expansion.
                The output tensor is of shape `(len(phi), num_spherical)`.
        """
        l_values = torch.arange(self.num_spherical)
        sph_harm_values = sph_harm(0, l_values[:, None], 0, phi).real
        return sph_harm_values.T


ANGULAR_FUNCTIONS = {
    "sph_harm": RealSphHarmBasis,
    "gaussian": GaussianBasis,
}


# TODO use envelope and expansion factories and use the default values if not provided here
class AngularBasisExpansion(torch.nn.Module):
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
    def forward(self, dist: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        rbf = self.radial_basis(dist)  # (num_edges, num_radial)
        abf = self.angular_basis(phi)  # (num_triplets, num_spherical)

        # ! Only work if len(dist) == len(phi)
        return rbf[:, None, :] * abf[:, :, None]  # (num_triplets, num_spherical, num_radial)
