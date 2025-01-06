from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.core.structure import FileFormats
from torch_geometric.data import Data


def get_cosine_angles(
    struct: Structure,
    i_indices: torch.Tensor,
    j_neighbors: torch.Tensor,
    k_neighbors: torch.Tensor,
) -> torch.Tensor:
    """Compute the cosine of the angles between the vectors formed by the central atom and its
    neighbors.

    Args:
        struct: A pymatgen structure object.
        i_indices: Indices of the central atoms.
        j_neighbors: Indices of the j neighbors.
        k_neighbors: Indices of the k neighbors.

    Returns:
        A tensor of shape (num_atoms, k) containing the cosine of the angles.
    """
    # Get coordinates of the central atoms, j neighbors, and k neighbors
    struct_coords = np.array([site.coords for site in struct], dtype=np.float32)
    central_coords = torch.from_numpy(struct_coords[i_indices])
    j_coords = torch.from_numpy(struct_coords[j_neighbors])
    k_coords = torch.from_numpy(struct_coords[k_neighbors])

    # Compute vectors
    v1 = j_coords - central_coords
    v2 = k_coords - central_coords

    # Compute dot product and norms
    dot_product = (v1 * v2).sum(dim=1)
    v1_norm = v1.norm(dim=1)
    v2_norm = v2.norm(dim=1)

    # Compute cosine of the angles
    cos_angles = dot_product / (v1_norm * v2_norm)

    return cos_angles


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

    def __init__(self, k: int = 20, rcut: float = 7.5) -> None:
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
            centers_idx, neighbors_idx, _, distances = struct.get_neighbor_list(
                r=self.rcut + delta_rcut, exclude_self=True
            )

            counts = np.bincount(centers_idx, minlength=n_atoms)
            reached_knn[counts >= self.k] = 1

            delta_rcut += 0.1 * self.rcut

        # Create the k-nearest neighbors index in a safe way
        # (i.e. if the number of neighbors is larger than k)
        knn_idx = np.zeros((n_atoms, self.k), dtype=int)
        for i in range(n_atoms):
            idx_i = np.where(centers_idx == i)[0]

            if len(idx_i) == self.k:
                knn_idx[i] = idx_i
            else:
                knn_mask = np.argpartition(distances[idx_i], self.k)[: self.k]
                knn_idx[i] = idx_i[knn_mask]

        # Only keep the k-nearest neighbors data
        knn_idx = knn_idx.flatten()
        centers_idx = torch.from_numpy(centers_idx[knn_idx])
        neighbors_idx = torch.from_numpy(neighbors_idx[knn_idx])
        distances = torch.from_numpy(distances[knn_idx].astype(np.float32))

        # Convert to PyG format
        edge_index = torch.vstack((centers_idx, neighbors_idx))

        # Angle cosine computation
        # Create indices for all combinations of j and k
        neighbor_indices = neighbors_idx.view(n_atoms, self.k)
        j_indices, k_indices = torch.triu_indices(self.k, self.k, offset=1)

        # Expand indices to match the number of atoms
        i_indices = torch.arange(n_atoms).repeat_interleave(len(j_indices))

        # Get the corresponding neighbor indices
        j_neighbors = neighbor_indices[:, j_indices].reshape(-1)
        k_neighbors = neighbor_indices[:, k_indices].reshape(-1)

        angles = get_cosine_angles(struct, i_indices, j_neighbors, k_neighbors)

        # Create the angle_cos tensor and fill it with computed angles
        angle_cos = torch.zeros(n_atoms, self.k, self.k, dtype=torch.float32)
        angle_cos[:, j_indices, k_indices] = angles.view(n_atoms, -1)
        angle_cos = angle_cos.view(n_atoms * self.k, self.k)

        return edge_index, distances, angle_cos

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
        struct = self._to_pymatgen_struct(struct, fmt=fmt)  #! 31% of runtime

        edge_index, edge_distances, angle_cos = self._get_graph_data(struct)

        data = Data(
            num_nodes=len(struct),
            pos=torch.tensor(struct.cart_coords, dtype=torch.float),
            edge_index=edge_index,
            edge_dist=edge_distances,
            angle_cos=angle_cos,
        )

        return data
