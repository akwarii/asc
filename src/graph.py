import io
from collections.abc import Callable
from functools import partial
from pathlib import Path

try:
    import faiss
    import faiss.contrib.torch_utils  # ignore
except ImportError:
    faiss = None  # type: ignore

import torch
from ase import Atoms
from freud.box import Box
from freud.locality import AABBQuery
from line_profiler import profile
from ovito.data import DataCollection
from ovito.io import import_file
from torch import Tensor
from torch_geometric.data import Data

from src.typing import FileFormats
from src.utils import atomic_numbers

CENTRAL_CELL = 13


def _get_graph_method(n_atoms: int) -> tuple[str, torch.device]:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    method = "torch"

    if faiss is not None:
        is_faiss_gpu = hasattr(faiss, "StandardGpuResources")

        if is_faiss_gpu:
            if device.type == "cpu":
                method = "faiss_cpu"
            else:
                method = "faiss_gpu"
        else:
            method = "faiss_cpu"
            device = torch.device("cpu")

    # TODO add dry run to benchmark and user argument in parser
    if "faiss" in method and n_atoms < 2_000:
        method = "torch"

    return method, device


class KNNGraph:
    """Helper class for creating a k-nearest neighbors graph from periodic structures.

    For each atom in the structure, edges are created to the nearest `k` neighbors.
    Self-loops are not created but can be easily added later using PyTorch Geometric.
    Note that the created graph is undirected. As such, the maximum number of neighbors
    is automatically reduced to the number of atoms in the structure minus one. This is
    not a problem for space group classification as a periodic site gives exactly the same
    information as the equivalent non-periodic site (if the features are invariant to rotations).

    Args:
        k: Number of neighbors.
    """

    def __init__(self, k: int = 20, **kwargs) -> None:
        if k < 1:
            raise ValueError("The number of neighbors must be greater than 0.")

        self.k = k

    def _get_graph_data(self, atoms: Atoms) -> tuple[Tensor, Tensor, Tensor]:
        n_atoms = len(atoms)

        knn_method, knn_device = _get_graph_method(n_atoms)

        cart_coords = torch.as_tensor(atoms.positions, dtype=torch.float32, device=knn_device)
        lat = torch.as_tensor(atoms.cell.array, dtype=torch.float32, device=knn_device)

        # TODO handle non periodic directions
        shifts = torch.tensor(
            [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            dtype=torch.int32,
            device=knn_device,
        )
        shifts_cart = shifts.to(torch.float32) @ lat

        imgs_cart = cart_coords.unsqueeze(0) + shifts_cart.unsqueeze(1)
        pts = imgs_cart.reshape(-1, 3).contiguous()

        knn_map: dict[str, Callable] = {
            "faiss_cpu": partial(self._faiss_knn, use_faiss_gpu=False),
            "faiss_gpu": partial(self._faiss_knn, use_faiss_gpu=True),
            "torch": self._torch_knn,
        }

        squared_dist, neighbors_idx = knn_map[knn_method](cart_coords, pts)

        distances = torch.sqrt(squared_dist)

        img_id = neighbors_idx // n_atoms
        atom_id = neighbors_idx % n_atoms

        # Drop self (central atom in central image) ---
        mask = ~(
            (img_id == CENTRAL_CELL)
            & (atom_id == torch.arange(n_atoms, device=knn_device)[:, None])
        )
        distances = distances[mask].reshape(n_atoms, -1)[:, : self.k]
        atom_id = atom_id[mask].reshape(n_atoms, -1)[:, : self.k]
        img_id = img_id[mask].reshape(n_atoms, -1)[:, : self.k]

        # Offset to get the actual coordinates of the neighbors
        offset_cart = shifts[img_id].to(torch.float32) @ lat

        # Build edge index
        centers_idx = torch.arange(n_atoms).repeat_interleave(self.k).to(knn_device)
        neighbors_idx = atom_id.reshape(-1)
        distances = distances.reshape(-1)
        edge_index = torch.vstack((centers_idx, neighbors_idx))

        # Distance components for line graph angles
        neighbor_indices = neighbors_idx.view(n_atoms, self.k)

        j_indices, k_indices = torch.triu_indices(self.k, self.k, offset=1, device=knn_device)
        i_indices = torch.arange(n_atoms, device=knn_device).repeat_interleave(len(j_indices))

        j_neighbors = neighbor_indices[:, j_indices].reshape(-1)
        k_neighbors = neighbor_indices[:, k_indices].reshape(-1)

        j_offset = offset_cart[:, j_indices].reshape(-1, 3)
        k_offset = offset_cart[:, k_indices].reshape(-1, 3)

        central_coords = cart_coords[i_indices]
        j_coords = cart_coords[j_neighbors] + j_offset
        k_coords = cart_coords[k_neighbors] + k_offset

        # Concatenate distance components (ij | ik)
        x = torch.cat((j_coords - central_coords, k_coords - central_coords), dim=1)

        # Move to the appropriate device
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        if knn_device != device:
            x = x.to(device)
            edge_index = edge_index.to(device)
            distances = distances.to(device)

        return x, edge_index, distances

    def _faiss_knn(
        self, cart_coords: Tensor, pts: Tensor, *, use_faiss_gpu: bool = False
    ) -> tuple[Tensor, Tensor]:
        cart_coords = cart_coords.contiguous()

        if use_faiss_gpu:
            res = faiss.StandardGpuResources()  # type: ignore
            squared_dist, neighbors_idx = faiss.knn_gpu(res, cart_coords, pts, self.k + 1)  # type: ignore
        else:
            squared_dist, neighbors_idx = faiss.knn(cart_coords, pts, self.k + 1)  # type: ignore

        return squared_dist, neighbors_idx  # type: ignore

    def _torch_knn(self, cart_coords: Tensor, pts: Tensor) -> tuple[Tensor, Tensor]:
        a_norm = torch.sum(cart_coords**2, dim=1, keepdim=True)
        b_norm = torch.sum(pts**2, dim=1).view(1, -1)

        all_squared_dist = a_norm + b_norm - 2 * torch.mm(cart_coords, pts.t())

        squared_dist, neighbors_idx = torch.topk(
            all_squared_dist, k=self.k + 1, dim=1, largest=False, sorted=False
        )  # as k is small 'sorted=False' should be equivalent to 'sorted=True' for performance

        return squared_dist, neighbors_idx

    @staticmethod
    def to_ase_atoms(atoms_repr: str | Path | Atoms, fmt: str | None = None) -> Atoms:
        """Load an ASE Atoms object from either.

        - a file path (str or Path), ASE will detect format by extension
        - a string containing structure data (CIF, VASP POSCAR, XYZ, LAMMPS-data, ...)

        Parameters:
            struct_input: Either a file path or a string containing the structure.
            fmt: The format of the structure if `struct_input` is a string.

        Returns:
            atoms : ase.Atoms
        """
        from ase.io import read

        if isinstance(atoms_repr, Atoms):
            atoms = atoms_repr

        elif isinstance(atoms_repr, Path):
            if not atoms_repr.exists():
                raise ValueError(f"The file {atoms_repr} does not exist.")
            if fmt is None and atoms_repr.suffix == ".lmp":
                fmt = "lammps-data"
            atoms = read(atoms_repr, format=fmt)

        elif isinstance(atoms_repr, str):
            import os

            if os.path.isfile(atoms_repr):
                atoms = read(atoms_repr)  # type: ignore

            # If we have a string representation of the structure
            else:
                if fmt is None:
                    fmt = detect_format_from_str(atoms_repr)

                stream = io.StringIO(atoms_repr.strip())
                atoms = read(stream, format=fmt)  # type: ignore

        else:
            raise TypeError(f"Unsupported input type {type(atoms_repr)}")

        return atoms  # type: ignore

    def convert(self, atoms_repr: Atoms | str | Path, fmt: str | None = None) -> Data:
        """Convert a single atomic structure to a PyG Data object.

        Args:
            atoms_repr: A pymatgen structure or an object convertible to a pymatgen structure.
            fmt: The format of the input structure if it is a string.

        Returns:
            A PyG Data object with positions, edge index, distances and cosine of the angles.
        """
        atoms = self.to_ase_atoms(atoms_repr, fmt=fmt)

        x, edge_index, edge_attr = self._get_graph_data(atoms)

        data = Data(
            num_nodes=len(atoms),
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return data


class PeriodicKNN:
    """Test of a periodic knn using Freud."""

    def __init__(self, k: int = 20, **kwargs) -> None:
        if k < 1:
            raise ValueError("The number of neighbors must be greater than 0.")

        self.k = k

    @profile
    def _get_graph_data(self, atoms: DataCollection) -> tuple[Tensor, Tensor, Tensor]:
        # Map OVITO particle types to atomic numbers
        type_mapper = {t.id: atomic_numbers[t.name] for t in atoms.particles.particle_types.types}

        elements_id = torch.from_numpy(atoms.particles.particle_types[...]).long()

        max_type = int(elements_id.max().item())
        mapping = torch.zeros(max_type + 1, dtype=torch.long)

        for lmp_type, z in type_mapper.items():
            mapping[lmp_type] = z

        x = mapping[elements_id]

        # extract data from ASE Atoms object
        pos = atoms.particles.positions[...]
        cell_matrix = atoms.cell[...][:3, :3].T

        # Create Freud Box for PBC handling
        box = Box.from_matrix(cell_matrix)
        box.periodic = atoms.cell.pbc

        # Perfom knn query
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

    @staticmethod
    def _to_ovito(representation: str | Path | DataCollection | io.StringIO) -> DataCollection:
        if isinstance(representation, DataCollection):
            return representation

        if isinstance(representation, io.StringIO):
            representation.seek(0)

        elif isinstance(representation, str):
            if not Path(representation).exists():
                representation = io.StringIO(representation.strip())
                representation.seek(0)

        elif isinstance(representation, Path):
            if not representation.exists():
                raise ValueError(f"The file {representation} does not exist.")

        else:
            raise TypeError(f"Unsupported input type {type(representation)}")

        print(type(representation))
        atoms = import_file(representation).compute()  # type: ignore

        return atoms

    def convert(
        self, atoms_repr: str | Path | DataCollection | io.StringIO, fmt: str | None = None
    ) -> Data:
        """Convert a single atomic structure to a PyG Data object.

        Args:
            atoms_repr: A pymatgen structure or an object convertible to a pymatgen structure.
            fmt: The format of the input structure if it is a string.

        Returns:
            A PyG Data object with positions, edge index, distances and cosine of the angles.
        """
        atoms = self._to_ovito(atoms_repr)

        cell_obj = atoms.cell
        assert (
            cell_obj is not None
        ), "The input structure must have a cell defined for periodic knn."

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


def detect_format_from_str(atoms_str: str) -> FileFormats:
    """Detect the format of a structure given as a string.

    Args:
        atoms_str: A string containing the structure.

    Returns:
        The format of the structure.
    """
    text = atoms_str.strip()
    first_line = text.splitlines()[0].strip()

    if "_cell_length_a" in text and "_atom_site" in text:
        return "cif"
    elif first_line.startswith("POSCAR") or first_line.startswith("CONTCAR"):
        return "vasp"
    elif first_line.isdigit():
        return "xyz"
    elif first_line.startswith("ITEM:"):
        return "lammps-dump-text"
    elif "Masses" in text and "Atoms" in text:
        return "lammps-data"
    else:
        lines = text.splitlines()
        try:
            float(lines[1].split()[0])  # scaling factor
            if all(len(line.split()) == 3 for line in lines[2:5]):  # lattice
                return "vasp"
        except ValueError:
            pass

    raise ValueError("Cannot guess structure format from string content")
