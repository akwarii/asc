import torch
from functools import cached_property


class ExponentialEnvelope(torch.nn.Module):
    """
    Exponential envelope function that ensures a smooth cutoff,
    as proposed in Unke et al (2021).
    SpookyNet: Learning Force Fields with Electronic Degrees of Freedom
    and Nonlocal Effects.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        """Forward pass of the exponential envelope function.

        Args:
            d_scaled (torch.Tensor): The scaled distance tensor.
        """
        env = torch.exp(-torch.pow(d_scaled, 2) / ((1 - d_scaled) * (1 + d_scaled)))
        return torch.where(d_scaled < 1, env, torch.zeros_like(d_scaled))


class PolynomialEnvelope(torch.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff,
    as proposed in Gasteiger et al (2022).
    Directional Message Passing for Molecular Graphs (arXiv:2003.03123).

    Args:
        degree (int, optional): The degree of the polynomial envelope. Defaults to 5.
    """

    def __init__(self, degree: int | float = 5) -> None:
        super().__init__()

        if (
            not (isinstance(degree, int) or (isinstance(degree, float) and degree.is_integer()))
            or degree < 1
        ):
            raise ValueError(
                "The degree of the polynomial envelope must be an integer larger than 0."
            )

        self.degree = degree

    @cached_property
    def _coeffs(self) -> tuple[float]:
        d = float(self.degree)
        a = -(d + 1) * (d + 2) / 2
        b = d * (d + 2)
        c = -d * (d + 1) / 2
        return a, b, c

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        """Forward pass of the polynomial envelope function.

        Args:
            d_scaled (torch.Tensor): The scaled distance tensor.
        """
        a, b, c = self._coeffs
        env = (
            1
            + a * torch.pow(d_scaled, self.degree)
            + b * torch.pow(d_scaled, self.degree + 1)
            + c * torch.pow(d_scaled, self.degree + 2)
        )
        return torch.where(d_scaled < 1, env, torch.zeros_like(d_scaled))


class DummyEnvelope(torch.nn.Module):
    """
    Dummy envelope function that does not apply any envelope.
    Using this envelope function is equivalent to using no envelope,
    meaning the bases orthonormality is conserved.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        """Forward pass of the dummy envelope function.

        Args:
            d_scaled (torch.Tensor): The scaled distance tensor.
        """
        return torch.ones_like(d_scaled)
