from collections.abc import Iterable

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class BoxStrain(BaseTransform):
    """Applies random strain (scaling and/or shearing) to a graph structure.

    Args:
        std: Standard deviation for the strain components.
        std_range: Range to sample the standard deviation from for each graph.
        directions: Strain components to apply. Options are:
            - "all": Full 3D strain (default).
            - a combination of 'xx', 'yy', 'zz', 'xy', 'xz', 'yz'
        scale_positions: Whether to transform node positions.
        transform_edge_attr: Whether to transform edge attributes (if they are 3D vectors).
    """

    COMPONENT_MAP = {
        "xx": (0, 0),
        "yy": (1, 1),
        "zz": (2, 2),
        "yz": (1, 2),
        "xz": (0, 2),
        "xy": (0, 1),
    }

    def __init__(
        self,
        std: float | None = None,
        std_range: tuple[float, float] | None = None,
        directions: Iterable[str] | str = "all",
        scale_positions: bool = True,
        transform_edge_attr: bool = True,
    ) -> None:
        self.std = std
        self.std_range = std_range

        if directions == "all":
            self.directions = {"xx", "yy", "zz", "xy", "xz", "yz"}
        elif isinstance(directions, str):
            self.directions = {directions}
        else:
            self.directions = set(directions)

        self.scale_positions = scale_positions
        self.transform_edge_attr = transform_edge_attr
        self.validate()

    def validate(self) -> None:
        """Validates the input parameters."""
        if self.std is None and self.std_range is None:
            raise ValueError("Either std or std_range must be provided.")

        invalid = [c for c in self.directions if c not in self.COMPONENT_MAP]
        if invalid:
            raise ValueError(
                f"Invalid directions: {invalid}. Directions must be one of "
                f"{list(self.COMPONENT_MAP.keys())} or 'all'."
            )

    def _get_std(self) -> Tensor:
        """Returns a random standard deviation sampled from the specified range if provided, else
        use a fixed value.
        """
        if self.std_range is not None:
            return torch.empty(1).uniform_(self.std_range[0], self.std_range[1])
        return torch.tensor(self.std)

    def _build_deformation_matrix(self) -> Tensor:
        """Builds a 3x3 deformation matrix based on the specified strain components."""
        deformation_matrix = torch.eye(3)

        indices = [self.COMPONENT_MAP[d] for d in self.directions]
        rows = torch.tensor([idx[0] for idx in indices])
        cols = torch.tensor([idx[1] for idx in indices])

        noise = torch.randn(len(self.directions)) * self._get_std()
        deformation_matrix[rows, cols] += noise

        return deformation_matrix

    def _check_shear_limit(self, cell: Tensor) -> None:
        """Checks if the shear transformation has resulted in a tilt factor that exceeds half the
        box length. This is important to ensure that the periodic boundary conditions remain valid
        after shearing.

        In practice, we check if a tilt factor (e.g., xy) is greater than half the box length in
        the parallel direction (the first dimension of the tilt factor).
        """
        lengths = torch.linalg.norm(cell, dim=0)
        ref_lengths = torch.stack([lengths[0], lengths[0], lengths[1]])

        tilts = torch.abs(torch.stack([cell[0, 1], cell[0, 2], cell[1, 2]]))

        if torch.any(tilts > 0.5 * ref_lengths):
            raise RuntimeError(
                "Shear transformation resulted in a tilt factor exceeding half the box length. "
                "This indicates the 'std' is too high, leading to invalid periodic boundary "
                "conditions."
            )

    def forward(self, data: Data) -> Data:
        """Applies the box strain transformation to the input graph data.

        Args:
            data: A PyG Data object with a 'cell' attribute representing the simulation box.

        Returns:
            The transformed Data object with the sheared cell and optionally transformed positions
            and edge attributes.
        """
        if not hasattr(data, "cell") or data.cell is None:
            return data

        deformation_matrix = self._build_deformation_matrix()
        deformation_matrix = deformation_matrix.to(data.cell)

        # Apply deformation to the cell vectors
        new_cell = data.cell @ deformation_matrix.T
        self._check_shear_limit(new_cell)
        data.cell = new_cell

        if self.scale_positions and hasattr(data, "pos") and data.pos is not None:
            data.pos = torch.matmul(data.pos, deformation_matrix.T)

        # Optionally apply shear to edge attributes if they are 3D vectors (e.g. edge vectors)
        if self.transform_edge_attr and hasattr(data, "edge_attr") and data.edge_attr is not None:
            # Vectors in 3D space should be transformed by the shear matrix as well
            if data.edge_attr.ndim == 2 and data.edge_attr.shape[-1] == 3:
                data.edge_attr = torch.matmul(data.edge_attr, deformation_matrix.T)

            # Scalar edge adtributes (e.g. distances) should be recomputed based on the new
            # positions because shearing will change the L2 norm between atoms.
            elif data.edge_attr.ndim == 2 and data.edge_attr.shape[-1] == 1:
                raise NotImplementedError(
                    "Recomputing scalar edge attributes based on sheared positions is not "
                    "implemented yet."
                )

        return data

    def __repr__(self) -> str:
        s = f"std_range={self.std_range}" if self.std_range else f"std={self.std}"
        return f"{self.__class__.__name__}({s}, directions={self.directions})"
