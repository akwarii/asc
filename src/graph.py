import os
import tempfile
from pathlib import Path

import torch
from freud.box import Box
from freud.locality import AABBQuery
from ovito.data import DataCollection
from ovito.io import import_file
from torch import Tensor
from torch_geometric.data import Data

from src.typing import PathLike
from src.utils import atomic_numbers


def type_id_to_atomic_number(type_id: Tensor, type_mapping: dict[int, int]) -> Tensor:
    """Convert a tensor of type ids to atomic numbers using a mapping.

    Args:
        type_id: A tensor of shape (num_atoms,) containing the type ids of the atoms.
        type_mapping: A dictionary mapping type ids to atomic numbers.

    Returns:
        A tensor of shape (num_atoms,) containing the atomic numbers of the atoms.
    """
    max_type_id = int(type_id.max().item())
    mapping_tensor = torch.zeros(max_type_id + 1, dtype=torch.long)
    for t_id, z in type_mapping.items():
        mapping_tensor[t_id - 1] = z

    atomic_numbers = mapping_tensor[type_id]

    return atomic_numbers


def read_structure(representation: PathLike | DataCollection) -> DataCollection:
    """Read a structure from a string, file path or OVITO DataCollection and return it in OVITO's
    format.

    Args:
        representation: A string containing the structure, a file path to a structure file, or an
            OVITO DataCollection object.
    """
    if isinstance(representation, DataCollection):
        return representation

    if isinstance(representation, Path):
        if not representation.exists():
            raise ValueError(f"The file {representation} does not exist.")
        return import_file(representation).compute()

    if isinstance(representation, str):
        if os.path.isfile(representation):
            return import_file(representation).compute()

        # We have a string representation of the structure so we write it to a temporary file
        # since OVITO's import_file function does not support file-like objects such as StringIO.
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp:
            tmp.write(representation)
            tmp_path = Path(tmp.name)

        atoms = import_file(tmp_path).compute()
        tmp_path.unlink(missing_ok=True)
        return atoms

    raise TypeError(f"Unsupported input type {type(representation)}")


class PeriodicKNN:
    """Test of a periodic knn using Freud."""

    def __init__(self, k: int = 20, **kwargs) -> None:
        if k < 1:
            raise ValueError("The number of neighbors must be greater than 0.")

        self.k = k

    def _get_graph_data(self, atoms: DataCollection) -> tuple[Tensor, Tensor, Tensor]:
        # Map OVITO particle types to atomic numbers
        type_mapper = {t.id: atomic_numbers[t.name] for t in atoms.particles.particle_types.types}
        type_id = torch.from_numpy(atoms.particles.particle_types[...]).long()

        x = type_id_to_atomic_number(type_id, type_mapper)

        # Extract data from OVITO DataCollection object
        pos = atoms.particles.positions[...]
        cell_matrix = atoms.cell[...][:3, :3].T

        # Create Freud Box for PBC handling
        box = Box.from_matrix(cell_matrix)
        box.periodic = atoms.cell.pbc

        # Perform knn query
        nq = AABBQuery(box, pos)
        nlist = nq.query(pos, dict(num_neighbors=self.k, exclude_ii=True)).toNeighborList()

        # Build edge index and attributes
        q_idx = torch.from_numpy(nlist.query_point_indices.copy()).long()
        p_idx = torch.from_numpy(nlist.point_indices.copy()).long()

        edge_index = torch.stack([q_idx, p_idx])

        pos_t = torch.from_numpy(pos).float()
        diff_t = pos_t[p_idx] - pos_t[q_idx]
        wrapped_diff = box.wrap(diff_t.numpy())
        edge_attr = torch.from_numpy(wrapped_diff).float()

        return x, edge_index, edge_attr

    def convert(self, atoms_repr: PathLike | DataCollection) -> Data:
        """Convert a single atomic structure to a PyG Data object.

        Args:
            atoms_repr: An OVITO DataCollection or an object convertible to it.

        Returns:
            A PyG Data object with positions, edge index, distances and cosine of the angles.
        """
        atoms = read_structure(atoms_repr)

        cell_obj = atoms.cell
        assert cell_obj is not None, (
            "The input structure must have a cell defined for periodic knn."
        )

        pbc = torch.tensor(atoms.cell_.pbc).bool()
        cell = torch.from_numpy(atoms.cell_[...][:3, :3].T).float()
        positions = torch.from_numpy(atoms.particles_.positions_[...]).float()

        x, edge_index, edge_attr = self._get_graph_data(atoms)

        data = Data(
            num_nodes=atoms.particles.count,
            x=x,
            pos=positions,
            cell=cell,
            pbc=pbc,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return data
