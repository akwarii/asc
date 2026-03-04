import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class BoxScaling(BaseTransform):
    """Applies random box scaling to a graph structure.

    The box (cell matrix) is scaled by a factor sampled from a normal distribution centered at 1.0.
    Optionally, node positions can be scaled by the same factor, and edge attributes can be
    recomputed based on the scaled positions.

    Args:
        std: Standard deviation of the scaling factor distribution (centered at 1.0).
        std_range: A tuple specifying the range from which to uniformly sample the standard
            deviation for each graph. If provided, this takes precedence over the `std` argument.
        scale_positions: Whether to scale node positions by the same factor as the box.
        recompute_edge_attrs: Whether to recompute edge attributes after scaling positions.
    """

    def __init__(
        self,
        std: float | None = None,
        std_range: tuple[float, float] | None = None,
        scale_positions: bool = True,
        recompute_edge_attrs: bool = True,
    ) -> None:
        self.std = std
        self.std_range = std_range
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

    def _get_std(self) -> torch.Tensor:
        if self.std_range is not None:
            return torch.empty(1).uniform_(self.std_range[0], self.std_range[1])
        if self.std is not None:
            return torch.tensor(self.std)

        raise RuntimeError("This should never happen since we check for this in the constructor.")

    def forward(self, data: Data) -> Data:
        """Runs the transform."""
        # Sample three independent scaling factors from normal distribution centered at 1.0
        std = self._get_std()
        scale_factors = torch.randn(3) * std + 1.0

        # Scale the cell (each column represents an axis)
        if hasattr(data, "cell") and data.cell is not None:
            data.cell = data.cell * scale_factors

        # Optionally scale positions
        if self.scale_positions and hasattr(data, "pos") and data.pos is not None:
            data.pos = data.pos * scale_factors

        # Optionally scale edge attributes
        if self.recompute_edge_attrs and hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = data.edge_attr * scale_factors

        return data

    def __repr__(self) -> str:
        if self.std_range is not None:
            return f"{self.__class__.__name__}(std_range={self.std_range})"
        return f"{self.__class__.__name__}(std={self.std})"
