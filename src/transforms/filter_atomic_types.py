from collections.abc import Iterable, Sequence

import numpy as np
import torch
from ovito.data import DataCollection
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from src.graph import read_structure
from src.utils import atomic_numbers


class FilterAtomicTypes(BaseTransform):
    """Keep only selected atomic species in a raw structure or graph.

    When given an OVITO structure representation, the transform removes all particles whose
    chemical species are not listed in ``include`` before graph construction. When given a
    PyG graph, it removes the corresponding nodes from the graph.

    Args:
        include: Atomic species to keep, provided as one value or an iterable of values,
            using atomic symbols (for example
            ``["Zr", "O"]``) and/or atomic numbers (for example ``[40, 8]``).
        strict: If ``True``, raise an error when none of the requested species are present in a
            graph. If ``False``, return the original graph unchanged in that case.
    """

    def __init__(self, include: Iterable[str | int] | str | int, strict: bool = True) -> None:
        include_values: list[str | int]
        if isinstance(include, (str, int)):
            include_values = [include]
        else:
            include_values = list(include)

        if not include_values:
            raise ValueError("`include` must contain at least one atomic symbol or number.")

        self.strict = strict
        self.include_z = self._to_atomic_numbers(include_values)

    def _to_atomic_numbers(self, include: Sequence[str | int]) -> set[int]:
        include_z: set[int] = set()
        for value in include:
            if isinstance(value, bool):
                raise ValueError("Boolean values are not valid atomic numbers.")

            if isinstance(value, int):
                if value < 0:
                    raise ValueError("Atomic numbers in `include` must be non-negative.")
                include_z.add(value)
                continue

            symbol = value.strip()
            symbol = symbol[:1].upper() + symbol[1:].lower()

            if symbol not in atomic_numbers:
                raise ValueError(f"Unknown chemical symbol in `include`: {value}")
            include_z.add(atomic_numbers[symbol])

        return include_z

    def _node_mask(self, x: Tensor) -> Tensor:
        if x.ndim == 2 and x.size(-1) == 1:
            x = x.squeeze(-1)
        elif x.ndim != 1:
            raise ValueError(
                "FilterAtomicTypes expects `data.x` to be a 1D tensor of atomic numbers "
                "or a shape (num_nodes, 1) tensor."
            )

        include = torch.tensor(sorted(self.include_z), dtype=torch.long, device=x.device)
        return torch.isin(x.long(), include)

    def _filter_structure(self, data: DataCollection) -> DataCollection:
        """Remove all atoms whose types are not listed in ``include``."""
        if data.particles is None:
            raise ValueError("FilterAtomicTypes requires a particle structure to be present.")

        particles = data.particles
        particle_types = particles.particle_types
        if particle_types is None:
            raise ValueError("FilterAtomicTypes requires a typed `Particle Type` property.")

        type_ids = np.asarray(particles.particle_types[...], dtype=np.int64)
        selected_type_ids = {
            particle_type.id
            for particle_type in particle_types.types
            if atomic_numbers.get(particle_type.name) in self.include_z
        }
        type_mask = np.isin(type_ids, sorted(selected_type_ids))

        if not bool(type_mask.any()):
            if self.strict:
                requested_ids = sorted(self.include_z)
                raise RuntimeError(
                    "No requested atomic species found in structure. "
                    f"Requested atomic numbers: {requested_ids}."
                )
            return data

        filtered = data.clone()
        filtered.particles_.delete_elements(~type_mask)

        return filtered

    def _filter_graph(self, data: Data) -> Data:
        """Remove all nodes whose atomic numbers are not listed in ``include``."""
        if not hasattr(data, "x") or data.x is None:
            raise ValueError("FilterAtomicTypes requires `data.x` to be present.")

        mask = self._node_mask(data.x)

        if not bool(mask.any()):
            if self.strict:
                requested = sorted(self.include_z)
                raise RuntimeError(
                    "No requested atomic species found in graph. "
                    f"Requested atomic numbers: {requested}."
                )
            return data

        return data.subgraph(mask)

    def forward(self, data: DataCollection | Data | str) -> DataCollection | Data:
        """Apply atom-type filtering to either a raw structure or a PyG graph."""
        if isinstance(data, Data):
            return self._filter_graph(data)

        if not isinstance(data, DataCollection):
            data = read_structure(data)

        return self._filter_structure(data)

    def __repr__(self) -> str:
        requested = sorted(self.include_z)
        return f"{self.__class__.__name__}(include_z={requested}, strict={self.strict})"
