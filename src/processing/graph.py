from collections.abc import Generator, Sequence
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data
from torch_geometric.utils import cumsum

from src.utils.typing import PathLike, SliceDictType


class KNNGraph:
    """Helper class for creating a k-nearest neighbors graph from periodic structures.

    For each atom in the structure, edges are created to the nearest `k` neighbors.
    Self-loops are not created but can be easily added later using PyTorch Geometric.
    Note that the created graph is undirected. As such, the maximum number of neighbors
    is automatically reduced to the number of atoms in the structure minus one. This is
    not a problem for space group classification as a periodic site gives exactly the same
    information as the equivalent non-periodic site (if the features are invariant to rotations).

    Args:
        k (int): Number of neighbors.
        rcut (float): Cutoff radius in Angstroms to search for neighbors.
    """

    def __init__(self, k: int = 20, rcut: float = 10.0) -> None:
        assert k > 0, "k must be greater than 0"
        assert rcut > 0, "rcut must be greater than 0"

        self.k = k
        self.rcut = rcut

    def _get_graph_data(self, struct: Structure) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a nearest neighbor search and returns edge index, distances.

        The number of neighbors is determined by the `k` attribute.
        However, if the number of atoms in the unit cell is smaller than `k`, the
        number of neighbors will be reduced to the number of atoms minus one. This is
        not a problem for space group classification as taking `k=n-1` neighbors gives
        all the symmetry information contained in the unit cell.

        Args:
            struct (pymatgen.core.Structure): A pymatgen structure object.

        Returns:
            edge_index (torch.LongTensor): A tensor of shape (2, num_edges) where each column
                represents an edge between two atoms.
            edge_distances (torch.FloatTensor): A tensor of shape (num_edges,) containing the
                distances between atoms in the edge_index tensor.
        """
        centers_idx, neighbors_idx, _, distances = struct.get_neighbor_list(
            r=self.rcut, exclude_self=True
        )

        _k = self.k if self.k < len(struct) - 1 else len(struct) - 1

        knn_idx = []
        for i in range(len(struct)):
            idx_i = (centers_idx == i).nonzero()[0]

            # TODO find a clever way to handle this. For now, let's just ignore the "problem".
            if len(idx_i) < _k:
                # raise ValueError(
                #     f"Atom {i} has less than {_k} neighbors ({len(idx_i)}/{_k}). Try to increase the cutoff radius."
                # )
                pass

            idx_sorted = np.argsort(distances[idx_i])[:_k]
            knn_idx.append(idx_i[idx_sorted])

        # Only keep the k-nearest neighbors
        knn_idx = np.concatenate(knn_idx)
        centers_idx = centers_idx[knn_idx]
        neighbors_idx = neighbors_idx[knn_idx]
        distances = distances[knn_idx]

        # Convert to PyG format
        edge_index = torch.LongTensor(np.vstack((centers_idx, neighbors_idx)))
        edge_distances = torch.FloatTensor(distances)

        # Computing bond angles cosines
        angle_cos = torch.zeros([edge_distances.size(0), self.k - 1])
        for ij, pair in enumerate(edge_index.T) :
            central, neigh1 = [int(idx) for idx in pair]
            knn_ij = torch.where(
                (edge_index[0] == central) & # Same central node
                (edge_index[1] != neigh1)    # All other neighbours
            )[0]
            rij_2 = torch.pow(edge_distances[ij],2)
            for neigh2 in edge_index[1,knn_ij] :
                knn_ik = torch.where(
                    (edge_index[0] == central) & # Same central node
                    (edge_index[1] != neigh2)    # All other neighbours
                )[0]
                # ij is the actual index of the pair (central, neigh1)
                # let's have ik for the pair (central, neigh2)
                for ik in [
                    _ik.item() for _ik in torch.where(
                        (edge_index[0] == central) &
                        (edge_index[1] == neigh2)
                    )[0]
                ] :
                    k = torch.where(knn_ij == ik)[0].item()
                    j = torch.where(knn_ik == ij)[0].item()
                    if angle_cos[ij, k] !=0 and angle_cos[ik,j] != 0 : continue # Triplet already done
                    ################################ NOTE ################################
                    # [DB] Al-Kashi theorem ::
                    #   cos(\alpha) = (r_ij^2 + r_ik^2 - r_jk^2) / (2*r_ij*r_ik)
                    #      (where i is the central atom)
                    # can not be computed as in many cases j and k are not neighbors, and
                    # their distance is not stored anywhere.
                    # We have to rely on pymatgen.Structure built-in get_angle() instead.
                    ############################ END OF NOTE #############################
                    cos_a = np.cos(
                        struct.get_angle(central, neigh1, neigh2)
                    )
                    angle_cos[ij, k] = cos_a
                    angle_cos[ik, j] = cos_a

        return edge_index, edge_distances, angle_cos

    @staticmethod
    def _to_pymatgen_struct(struct_repr: str | Path) -> Structure:
        if isinstance(struct_repr, str):
            struct = Structure.from_str(struct_repr, fmt="poscar")
        elif isinstance(struct_repr, Path):
            if not struct_repr.is_file():
                raise FileNotFoundError(f"The file {struct_repr} does not exist.")
            struct = Structure.from_file(struct_repr)
        elif isinstance(struct_repr, Structure):
            struct=struct_repr # modified by DB (previously pass)
        else:
            raise ValueError("The input must be a pymatgen structure object, a string or a path.")

        return struct

    def convert(
        self,
        struct: Structure,
        mask_sites: torch.BoolTensor | None = None,
    ) -> Data:
        """Convert a single atomic structure to a graph.

        Args:
            struct (Structure | str | Path): A pymatgen structure or an object convertible to a
                pymatgen structure. String must be in POSCAR format.
            mask (torch.BoolTensor, optional): A tensor of shape (num_atoms,) used to mask atoms.
                The masked atoms will not be used during training or inference. Defaults to None.

        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with positions, cell matrix,
                edge index, edge distances and an optional mask.
        """
        if isinstance(struct, str) or isinstance(struct, Path): # if added by DB
            struct = self._to_pymatgen_struct(struct)

        edge_index, edge_distances, angle_cos = self._get_graph_data(struct)
        num_nodes = len(struct)

        if mask_sites is not None:
            assert (
                len(mask_sites.shape) == 1 and len(mask_sites) == num_nodes
            ), "The number of tags must match the number of atoms in the structure."
            assert mask_sites.dtype == torch.bool, "The mask must be a boolean tensor."

        # put the minimum data in torch geometric data object
        data = Data(
            num_nodes=num_nodes,
            pos=torch.FloatTensor(struct.cart_coords),
            cell=torch.FloatTensor(struct.lattice.matrix.copy()),
            edge_index=edge_index,
            edge_dist=edge_distances,
            angle_cos=angle_cos, # TODO: Change rattle ?
            mask=mask_sites,
        )

        return data

    def batch_conversion(
        self,
        structs: Sequence[Structure | str | Path],
        mask_struct_sites: dict[int, torch.BoolTensor] | None = None,
        progress_bar: bool = True,
    ) -> Generator[Data, None, None]:
        """Convert a list of atomic structures to a list of graphs.

        Args:
            structs (list[pymatgen.core.Structure]): A list of pymatgen structure objects or of objects
                convertible to a pymatgen structure. Note that providing a list of paths or strings is
                slower but can save a significant amount of memory for large datasets. Strings must be
                in POSCAR format.
            mask_struct_sites (tuple, optional): A dict where the key is the index of the struct to which the
                mask should be applied. The masked atoms will not be used during training or inference.
                Defaults to None.

        Returns:
            data (Generator[torch_geometric.data.Data]): A generator of torch geometric data objects. Each data
                object contains information on the positions, cell matrix, edge index, edge distances and an
                optional mask.
        """
        if progress_bar:
            from tqdm import tqdm

            pbar = tqdm(total=len(structs), desc="Converting structures to graphs")

        if mask_struct_sites is None:
            mask_struct_sites = dict()

        for i, struct in enumerate(structs):
            mask = mask_struct_sites.get(i)

            graph = self.convert(struct, mask)

            if progress_bar:
                pbar.update(1)

            yield graph

        if progress_bar:
            pbar.close()

    def convert_and_save(
        self,
        structs: Sequence[Structure | str | Path],
        path: PathLike,
        chunk_size: int | None = None,
        mask_struct_sites: dict[int, torch.BoolTensor] | None = None,
        progress_bar: bool = True,
    ) -> None:
        """Convert a list of atomic structures to a list of graphs and save them to a file. Note
        that the whole list of graphs must fit in memory. If the dataset is too large, consider
        using the `batch_conversion` method and saving the graphs in smaller chunks. Once the data
        are saved, they can be loaded using the `torch.load` function.

        Args:
            structs (list[pymatgen.core.Structure]): A list of pymatgen structure objects or of objects
                convertible to a pymatgen structure. Note that providing a list of paths or strings is
                slower but can save a significant amount of memory for large datasets. Strings must be
                in POSCAR format.
            path (PathLike): The path to the file where the graphs will be saved.
            mask_struct_sites (tuple, optional): A dict where the key is the index of the struct to which the
                mask should be applied. The masked atoms will not be used during training or inference.
                Defaults to None.
        """
        # TODO: Implement chunking
        if chunk_size is not None:
            raise NotImplementedError("Saving graphs in chunks is not yet implemented.")

        graphs = list(self.batch_conversion(structs, mask_struct_sites, progress_bar))
        data, slices = self.collate(graphs)

        torch.save((data.to_dict(), slices), path) 

    @staticmethod
    def collate(data_list: Sequence[Data]) -> tuple[Data, SliceDictType | None]:
        """Simplified version of `torch_geometric.data.collate` function.

        Collates a list of `data` objects into a single object of type `cls`.
        `collate` can handle both homogeneous and heterogeneous data objects by
        individually collating all their stores.
        In addition, `collate` can handle nested data structures such as
        dictionaries and lists.
        """

        if not isinstance(data_list, (list, tuple)):
            data_list = list(data_list)

        if not data_list:
            raise ValueError("data_list is empty.")

        if len(data_list) == 1:
            # return data_list[0][0], None # DB added [0]
            return data_list[0], None

        # Create empty stores
        out = data_list[0][0].__class__() # DB added [0]
        out.stores_as(data_list[0][0]) # DB added [0]

        # Group storage objects of every data object by key
        key_to_stores = {store._key: [] for store in data_list[0][0].stores} # DB added [0]
        for data in data_list: 
            for store in data[0].stores: # DB added [0]
                key_to_stores[store._key].append(store)

        # Iterate over each list of storage objects and recursively collate all its attributes.
        # `slice_dict` stores a compressed index representation of each attribute
        #  and is needed to re-construct individual elements from mini-batches.
        slice_dict: SliceDictType = {}
        for out_store in out.stores:
            key = out_store._key
            stores = key_to_stores[key]

            for attr in stores[0].keys():
                values = [store[attr] for store in stores]

                # `num_nodes` needs to be summed up and not concatenated.
                if attr == "num_nodes":
                    out_store._num_nodes = values
                    out_store.num_nodes = sum(values)
                    continue

                # Concatenate a list of `torch.Tensor` along `cat_dim`.
                # and appropriately take care of incrementing elements.
                cat_dim = data_list[0][0].__cat_dim__(attr, values[0], stores[0])# DB added [0]
                sizes = torch.tensor([value.size(cat_dim or 0) for value in values])
                slices = cumsum(sizes)
                value = torch.cat(values, dim=cat_dim or 0)

                out_store[attr] = value
                slice_dict[attr] = slices

        return out, slice_dict
