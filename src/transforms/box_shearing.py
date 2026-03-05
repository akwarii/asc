from collections.abc import Iterable

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class BoxShearing(BaseTransform):
    """Applies random box shearing (shear strain) to a graph structure.

    The box (cell matrix) is sheared by applying a shear transformation matrix with random
    shear components sampled from a normal distribution. Optionally, node positions and edge
    attributes can be transformed by the same shear matrix.

    Args:
        std: Standard deviation of the shear strain distribution.
        std_range: A tuple specifying the range from which to uniformly sample the standard
            deviation for each graph. If provided, this takes precedence over the `std` argument.
        shear_components: List of shear components to apply. Options are 'xy', 'xz', 'yz'.
            Default is ['xy', 'xz', 'yz']. Each component is sheared independently with the
            same standard deviation.
        scale_positions: Whether to apply the shear transformation to node positions.
        recompute_edge_attrs: Whether to apply the shear transformation to edge attributes.
    """

    def __init__(
        self,
        std: float | None = None,
        std_range: tuple[float, float] | None = None,
        shear_components: Iterable[str] | None = None,
        scale_positions: bool = True,
        recompute_edge_attrs: bool = True,
    ) -> None:
        self.std = std
        self.std_range = std_range
        self.shear_components = (
            set(shear_components) if shear_components is not None else {"xy", "xz", "yz"}
        )
        self.scale_positions = scale_positions
        self.recompute_edge_attrs = recompute_edge_attrs

        self.validate()

    def validate(self) -> None:
        """Validates the input parameters."""
        if self.std is None and self.std_range is None:
            raise ValueError("Either std or std_range must be provided.")
        if self.std is not None and self.std < 0.0:
            raise ValueError("The standard deviation must be non-negative.")
        if self.std_range is not None and (self.std_range[0] < 0.0 or self.std_range[1] < 0.0):
            raise ValueError("The standard deviation range must be non-negative.")

        valid_components = {"xy", "xz", "yz"}
        if not self.shear_components.issubset(valid_components):
            raise ValueError(f"Shear components must be subset of {valid_components}.")

    def _get_std(self) -> Tensor:
        if self.std_range is not None:
            return torch.empty(1).uniform_(self.std_range[0], self.std_range[1])
        if self.std is not None:
            return torch.tensor(self.std)

        raise RuntimeError("This should never happen since we check for this in the constructor.")

    def _build_shear_matrix(self, *, dtype: torch.dtype, device: torch.device) -> Tensor:
        """Builds a 3x3 shear transformation matrix.

        Returns:
            A 3x3 shear transformation matrix.
        """
        shear_matrix = torch.eye(3, dtype=dtype, device=device)
        std = self._get_std()

        # Apply shear components
        if "xy" in self.shear_components:
            shear_matrix[0, 1] = torch.randn((), dtype=dtype, device=device) * std
        if "xz" in self.shear_components:
            shear_matrix[0, 2] = torch.randn((), dtype=dtype, device=device) * std
        if "yz" in self.shear_components:
            shear_matrix[1, 2] = torch.randn((), dtype=dtype, device=device) * std

        return shear_matrix

    def _check_shear_limit(self, cell: Tensor, shear_matrix: Tensor) -> None:
        """Checks that face offsets do not exceed half-box length in shearing direction.

        Returns:
            True if all shear components are valid, False otherwise.
        """
        lengths = torch.linalg.norm(cell, dim=0)
        component_to_indices = {
            "xy": (0, 1),
            "xz": (0, 2),
            "yz": (1, 2),
        }

        # Extract indices for active components
        indices = [component_to_indices[comp] for comp in self.shear_components]
        if not indices:
            return

        i_vals = torch.tensor([idx[0] for idx in indices], device=cell.device, dtype=torch.long)
        j_vals = torch.tensor([idx[1] for idx in indices], device=cell.device, dtype=torch.long)

        # Extract shear coefficients using fancy indexing
        shear_coeffs = torch.abs(shear_matrix[i_vals, j_vals])

        # Compute offsets: offset[k] = shear_coeff[k] * lengths[j_vals[k]]
        offsets = shear_coeffs * lengths[j_vals]

        # Compute max offsets: max_offset[k] = 0.5 * lengths[i_vals[k]]
        max_offsets = 0.5 * lengths[i_vals]

        # Check if all offsets are valid
        valid = torch.all(offsets <= max_offsets)
        if not valid:
            raise RuntimeError(
                "Shear transformation would result in face offsets exceeding half-box length. "
                "This indicates that your chosen standard deviation is way "
                "too large for the box size."
            )

    def forward(self, data: Data) -> Data:
        """Runs the transform."""
        if not hasattr(data, "cell") or data.cell is None:
            return data

        # Build shear transformation matrix
        shear_matrix = self._build_shear_matrix(dtype=data.cell.dtype, device=data.cell.device)

        self._check_shear_limit(data.cell, shear_matrix)

        # Apply shear to cell
        data.cell = torch.matmul(data.cell, shear_matrix.T)

        # Optionally apply shear to positions
        if self.scale_positions and hasattr(data, "pos") and data.pos is not None:
            data.pos = torch.matmul(data.pos, shear_matrix.T)

        # Optionally apply shear to edge attributes
        if self.recompute_edge_attrs and hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = torch.matmul(data.edge_attr, shear_matrix.T)

        return data

    def __repr__(self) -> str:
        std_str = (
            f"std_range={self.std_range}" if self.std_range is not None else f"std={self.std}"
        )
        return f"{self.__class__.__name__}({std_str}, shear_components={self.shear_components}, "
