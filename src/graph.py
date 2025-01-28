from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.core.structure import FileFormats
from torch_geometric.data import Data


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
        rcut: Cutoff radius in Angstroms to search for neighbors.
    """

    def __init__(self, k: int = 20, rcut: float = 7.5, **kwargs) -> None:
        if k < 1:
            raise ValueError("The number of neighbors must be greater than 0.")
        if rcut <= 0.0:
            raise ValueError("The cutoff radius must be greater than 0.0")

        self.k = k
        self.rcut = rcut

    # TODO docstring
    def _get_graph_data(
        self, struct: Structure
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a nearest neighbor search and returns edge index, distances.

        The number of neighbors is determined by the `k` attribute.
        However, if the number of atoms in the unit cell is smaller than `k`, the
        number of neighbors will be reduced to the number of atoms minus one. This is
        not a problem for space group classification as taking `k=n-1` neighbors gives
        all the symmetry information contained in the unit cell.

        Args:
            struct: A pymatgen structure object.

        Returns:
            edge_index (torch.LongTensor): A tensor of shape (2, num_edges) where each column
                represents an edge between two atoms.
            edge_distances (torch.FloatTensor): A tensor of shape (num_edges,) containing the
                distances between atoms in the edge_index tensor.
        """
        n_atoms = len(struct)
        reached_knn = np.zeros(n_atoms, dtype=bool)
        delta_rcut = 0.0

        # Find the k-nearest neighbors
        while not np.all(reached_knn):
            all_centers_idx, all_neighbors_idx, all_offset, all_distances = (
                struct.get_neighbor_list(r=self.rcut + delta_rcut, exclude_self=True)
            )

            counts = np.bincount(all_centers_idx, minlength=n_atoms)
            reached_knn[counts >= self.k] = 1

            delta_rcut += 0.1 * self.rcut

        # Create the k-nearest neighbors index in a safe way
        # (i.e. if the number of neighbors is larger than k, it can't be smaller)
        knn_idx = np.zeros((n_atoms, self.k), dtype=int)
        for i in range(n_atoms):
            idx_i = np.where(all_centers_idx == i)[0]

            if len(idx_i) == self.k:
                knn_idx[i] = idx_i
            else:
                knn_mask = np.argpartition(all_distances[idx_i], self.k)[: self.k]
                knn_idx[i] = idx_i[knn_mask]

        # Only keep the k-nearest neighbors data (and jump to torch.Tensors as well)
        knn_idx = knn_idx.flatten()
        centers_idx: torch.Tensor = torch.from_numpy(all_centers_idx[knn_idx])
        neighbors_idx: torch.Tensor = torch.from_numpy(all_neighbors_idx[knn_idx])
        distances: torch.Tensor = torch.from_numpy(all_distances[knn_idx].astype(np.float32))
        offset: torch.Tensor = torch.from_numpy(all_offset[knn_idx].astype(np.float32))

        # Convert to PyG format
        edge_index = torch.vstack((centers_idx, neighbors_idx))

        # Distance components (required for LineGraph angles computations)
        cell = torch.from_numpy(struct.lattice.matrix.astype(np.float32))
        struct_coords = struct.cart_coords.astype(np.float32)
        neighbor_indices = neighbors_idx.view(n_atoms, self.k)
        offset = offset.view(n_atoms, self.k, 3)
        j_indices, k_indices = torch.triu_indices(self.k, self.k, offset=1)

        # Expand indices to match the number of atoms
        i_indices = torch.arange(n_atoms).repeat_interleave(len(j_indices))

        # Get the corresponding neighbor indices (and offsets)
        j_neighbors = neighbor_indices[:, j_indices].reshape(-1)
        k_neighbors = neighbor_indices[:, k_indices].reshape(-1)

        # Coordinates
        central_coords = torch.from_numpy(struct_coords[i_indices])
        j_coords = torch.from_numpy(struct_coords[j_neighbors]) + torch.matmul(
            offset[:, j_indices].reshape(j_neighbors.size()[0], 3), cell
        )
        k_coords = torch.from_numpy(struct_coords[k_neighbors]) + torch.matmul(
            offset[:, k_indices].reshape(k_neighbors.size()[0], 3), cell
        )

        # Actual distance components - we do not use `torch.stack` because
        # adding an extra dimension (to use x[0], x[1] for ij, ik) would
        # cause mismatches in `collate` where "Sizes of tensors must match
        # except in dimension 0.". Instead, we use `torch.cat` to avoid this
        # extra dimension (and use x[:,:3], x[:,3:] for ij, ik).
        x = torch.cat((j_coords - central_coords, k_coords - central_coords), dim=1)

        return x, edge_index, distances

    @staticmethod
    def _to_pymatgen_struct(
        struct_repr: Structure | str | Path, fmt: FileFormats = "poscar"
    ) -> Structure:
        if isinstance(struct_repr, Structure):
            struct = struct_repr
        elif isinstance(struct_repr, str):
            struct = Structure.from_str(struct_repr, fmt=fmt)
        elif isinstance(struct_repr, Path) and not struct_repr.is_file():
            if not struct_repr.is_file():
                raise FileNotFoundError(f"The file {struct_repr} does not exist.")
            struct = Structure.from_file(struct_repr)
        else:
            raise ValueError("The input must be a pymatgen structure object, a string or a path.")

        return struct

    def convert(self, struct: Structure | str | Path, fmt: FileFormats = "poscar") -> Data:
        """Convert a single atomic structure to a PyG Data object.

        Args:
            struct: A pymatgen structure or an object convertible to a pymatgen structure.
            fmt: The format of the input structure if it is a string.

        Returns:
            A PyG Data object with positions, edge index, distances and cosine of the angles.
        """
        struct = self._to_pymatgen_struct(struct, fmt=fmt)

        x, edge_index, edge_distances = self._get_graph_data(struct)

        data = Data(
            num_nodes=len(struct),
            x=x,
            edge_index=edge_index,
            edge_attr=edge_distances,
        )

        return data
