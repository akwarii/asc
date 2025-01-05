from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
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

    def __init__(self, k: int = 20, rcut: float = 10.0) -> None:
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
        reached_knn = np.zeros(len(struct))
        delta_rcut = 0.0

        # TODO use a kdtree ?
        while not np.all(reached_knn):
            centers_idx, neighbors_idx, _, distances = struct.get_neighbor_list(
                r=self.rcut + delta_rcut, exclude_self=True
            )

            knn_idx = []
            for i in range(len(struct)):
                idx_i = (centers_idx == i).nonzero()[0]

                if len(idx_i) < self.k:
                    delta_rcut += 0.05 * self.rcut
                else:
                    reached_knn[i] = 1

                idx_sorted = np.argsort(distances[idx_i])[: self.k]
                knn_idx.append(idx_i[idx_sorted])

        # Only keep the k-nearest neighbors
        knn_idx = np.concatenate(knn_idx)
        centers_idx = centers_idx[knn_idx]
        neighbors_idx = neighbors_idx[knn_idx]
        distances = distances[knn_idx]

        # Convert to PyG format
        edge_index = torch.LongTensor(np.vstack((centers_idx, neighbors_idx)))
        edge_distances = torch.FloatTensor(distances)

        # DB: Angle cosine computation, directly adapted from OG CEGANN
        # see /CEGANN/graph.py @ SetGraphFea
        m = len(struct)
        _nbr_idx = torch.reshape(torch.LongTensor(neighbors_idx), (m, self.k))
        bond = torch.reshape(edge_distances, (m, self.k))
        cart_coords = torch.Tensor(np.array([struct[i].coords for i in range(m)]))
        # FIXME error sometimes but not always here
        atom_nbr_fea = torch.Tensor(
            np.array([[struct[j].coords for j in _nbr_idx[i]] for i in range(m)])
        )
        centre_coords = cart_coords.unsqueeze(1).expand(m, self.k, 3)
        dxyz = atom_nbr_fea - centre_coords
        r = bond.unsqueeze(2)
        angle_cos = torch.matmul(dxyz, torch.swapaxes(dxyz, 1, 2)) / torch.matmul(
            r, torch.swapaxes(r, 1, 2)
        )
        angle_cos = angle_cos.flatten(0, 1)  # To fit into collate

        return edge_index, edge_distances, angle_cos

    @staticmethod
    def _to_pymatgen_struct(struct_repr: str | Path) -> Structure:
        if isinstance(struct_repr, str):
            struct = Structure.from_str(struct_repr, fmt="poscar")
        elif isinstance(struct_repr, Path):
            if not struct_repr.is_file():
                raise FileNotFoundError(f"The file {struct_repr} does not exist.")
            struct = Structure.from_file(struct_repr)
        else:
            raise ValueError("The input must be a pymatgen structure object, a string or a path.")

        return struct

    def convert(
        self,
        struct: Structure | str | Path,
        mask_sites: torch.BoolTensor | None = None,
    ) -> Data:
        """Convert a single atomic structure to a PyG Data object.

        Args:
            struct: A pymatgen structure or an object convertible to a pymatgen structure. String
                must be in POSCAR format.
            mask_sites: A tensor of shape (num_atoms,) used to mask atoms. The masked atoms will
            not be used during training or inference.

        Returns:
            A PyG Data object with positions, cell matrix, edge index, edge distances and an
                optional mask.
        """
        if isinstance(struct, str) or isinstance(struct, Path):  # if added by DB
            struct = self._to_pymatgen_struct(struct)

        edge_index, edge_distances, angle_cos = self._get_graph_data(struct)
        num_nodes = len(struct)

        # TODO change assert to raise error
        if mask_sites is not None:
            assert (
                len(mask_sites.shape) == 1 and len(mask_sites) == num_nodes
            ), "The number of tags must match the number of atoms in the structure."
            assert mask_sites.dtype == torch.bool, "The mask must be a boolean tensor."

        # put the minimum data in torch geometric data object
        data = Data(
            num_nodes=num_nodes,
            pos=torch.FloatTensor(struct.cart_coords),
            # cell=torch.FloatTensor(struct.lattice.matrix.copy()),
            edge_index=edge_index,
            edge_dist=edge_distances,
            angle_cos=angle_cos,  # TODO: Change rattle ?
            mask=mask_sites,
        )

        return data
