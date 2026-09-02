from collections.abc import Iterable

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


def _normalize_apply_to(apply_to: str | Iterable[str]) -> set[str]:
    """Normalize ``apply_to`` into a set of attribute names.

    A bare string is treated as a single attribute name instead of being split into chars.

    For example, "pos" is returned as a single attribute name, but would be treated as
    three attribute names {'p', 'o', 's'} without this normalization.

    If an iterable of strings is provided, it is converted to a set of attribute names.

    Args:
        apply_to (str or Iterable[str]): A string or an iterable of strings specifying which
            attributes to perturb.

    Returns:
        set[str]: A set of attribute names to perturb.
    """
    if isinstance(apply_to, str):
        names = [apply_to]
    else:
        try:
            names = list(apply_to)
        except TypeError as exc:
            raise TypeError(
                "apply_to must be a string or an iterable of strings"
                f", got {type(apply_to).__name__}"
            ) from exc

    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"apply_to entries must be strings, got {type(name).__name__}: {name!r}"
            )

    return set(names)


class RandomPerturbation(BaseTransform):
    """Applies random Gaussian noise to both node and edge features of a graph if they exist.

    Note that the edge attributes are only recomputed if the node positions are perturbed and the
    `recompute_edge_attrs` flag is set to True.

    Note:
        This transform being non-deterministic, it is intended to be used only during training. It
        is therefore automatically removed from the validation, testing and prediction dataloaders
        when created by the `LightningDataset`.

    Args:
        std: Standard deviation of the applied noise.
        std_range: A tuple specifying the range from which to uniformly sample the standard
        deviation for each graph. If provided, this takes precedence over the `std` argument.
        apply_to: A string or list of strings specifying which attributes to perturb.
        recompute_edge_attrs: Whether to recompute edge attributes after perturbing node positions.
    """

    def __init__(
        self,
        std: float | None = None,
        std_range: tuple[float, float] | None = None,
        apply_to: str | Iterable[str] = "pos",
        recompute_edge_attrs: bool = True,
    ) -> None:
        self.std = std
        self.std_range = std_range
        self.apply_to = _normalize_apply_to(apply_to)
        self.recompute_edge_attrs = recompute_edge_attrs

        self.validate()

    def validate(self) -> None:
        """Validates the input parameters."""
        if self.std is None and self.std_range is None:
            raise ValueError("Either std or std_range must be provided.")
        if self.std is not None and self.std < 0.0:
            raise ValueError("The standard deviation must be positive.")
        if self.std_range is not None:
            lo, hi = self.std_range
            if lo < 0.0 or hi < 0.0:
                raise ValueError("The standard deviation range must be positive.")
            if lo > hi:
                raise ValueError("The standard deviation range must satisfy lower <= upper.")

    def _get_std(self) -> Tensor:
        if self.std_range is not None:
            return torch.empty(1).uniform_(self.std_range[0], self.std_range[1])
        if self.std is not None:
            return torch.tensor(self.std)

        raise RuntimeError("This should never happen since we check for this in the constructor.")

    def _wrap(self, vectors: Tensor, cell: Tensor, pbc: Tensor) -> Tensor:
        """Applies MIC to a triclinic box with optional periodicity per dimension.

        Args:
            vectors: (N, 3) displacement vectors.
            cell: (3, 3) matrix where columns are [a, b, c].
            pbc: Boolean flags for (x, y, z) periodicity.
        """
        # Transform to fractional coordinates
        inv_box = torch.linalg.inv(cell)
        s = torch.matmul(vectors, inv_box.T)

        # Create a wrapping term for periodic dimensions
        wrap_term = torch.round(s) * pbc
        s_mic = s - wrap_term

        # Transform back to Cartesian
        r_mic = torch.matmul(s_mic, cell.T)

        return r_mic

    def forward(self, data: Data) -> Data:
        """Runs the transform."""
        for attr in sorted(self.apply_to):
            if hasattr(data, attr):
                attr_val = getattr(data, attr)
                if attr_val is not None:
                    noise = torch.randn_like(attr_val) * self._get_std()
                    setattr(data, attr, attr_val + noise)

        if self.recompute_edge_attrs:
            if "pos" in self.apply_to and hasattr(data, "pos") and data.pos is not None:
                assert data.edge_index is not None
                assert data.cell is not None

                new_edge_attr = data.pos[data.edge_index[1]] - data.pos[data.edge_index[0]]
                new_edge_attr = self._wrap(new_edge_attr, data.cell, data.pbc)

                data.edge_attr = new_edge_attr

        return data

    def __repr__(self) -> str:
        if self.std_range is not None:
            return f"{self.__class__.__name__}(std_range={self.std_range})"
        return f"{self.__class__.__name__}(stddev={self.std})"
