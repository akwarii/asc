from collections import defaultdict
from collections.abc import Generator, Sequence
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data
from torch_geometric.utils import cumsum

from src.typing import PathLike, SliceDictType


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
        # TODO change assert to raise error
        assert k > 0, "k must be greater than 0"
        assert rcut > 0, "rcut must be greater than 0"

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

        while not np.all(reached_knn):
            centers_idx, neighbors_idx, _, distances = struct.get_neighbor_list(
                r=self.rcut + delta_rcut, exclude_self=True
            )

            knn_idx = []
            for i in range(len(struct)):
                idx_i = (centers_idx == i).nonzero()[0]

                # TODO find a clever way to handle this. For now, let's just ignore the "problem".
                if len(idx_i) < self.k:
                    delta_rcut += 0.01 * self.rcut
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
            cell=torch.FloatTensor(struct.lattice.matrix.copy()),
            edge_index=edge_index,
            edge_dist=edge_distances,
            angle_cos=angle_cos,  # TODO: Change rattle ?
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
            structs: A list of pymatgen structure objects or of objects convertible to a pymatgen
                structure. Note that providing a list of paths or strings is slower but can save a
                significant amount of memory for large datasets. Strings must be in POSCAR format.
            mask_struct_sites: A dict where the key is the index of the struct to which the mask
                should be applied. The masked atoms will not be used during training or inference.
            progress_bar: Whether to display a progress bar.

        Yields:
        ------
            data: A generator of torch geometric data objects. Each data object contains
            information on the positions, cell matrix, edge index, edge distances and an optional
            mask.
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
            structs: A list of pymatgen structure objects or of objects convertible to a pymatgen
                structure. Note that providing a list of paths or strings is slower but can save a
                significant amount of memory for large datasets. Strings must be in POSCAR format.
            path: The path to the file where the graphs will be saved.
            chunk_size: The number of graphs to save in each chunk. If `None`, all graphs will be
                saved in a single file.
            mask_struct_sites: A dict where the key is the index of the struct to which the mask
                should be applied. The masked atoms will not be used during training or inference.
            progress_bar: Whether to display a progress bar.
        """
        # TODO: Implement chunking
        if chunk_size is not None:
            raise NotImplementedError("Saving graphs in chunks is not yet implemented.")

        # TODO add targets placeholder as they will be needed to collate
        # TODO however the real targets will be loaded from the dataset
        # TODO saving the graphs on the disk is just a way to avoid computing them during training
        graphs = list(self.batch_conversion(structs, mask_struct_sites, progress_bar))
        data, _, slices = self.collate(graphs)

        torch.save((data.to_dict(), slices), path)

    # TODO docstring
    @staticmethod
    def collate(data_list: Sequence[tuple[Data, int]]) -> tuple[Data, torch.Tensor, SliceDictType]:
        """Simplified version of `torch_geometric.data.collate` function.

        Collates a list of `data` objects into a single object of type `cls`.
        `collate` can handle both homogeneous and heterogeneous data objects by
        individually collating all their stores.
        In addition, `collate` can handle nested data structures such as
        dictionaries and lists.
        """
        if not isinstance(data_list, list | tuple):
            data_list = list(data_list)

        if not data_list:
            raise ValueError("data_list is empty.")

        if len(data_list) == 1:
            return data_list[0][0], torch.LongTensor(data_list[0][1]), None

        # Target values
        targets = torch.LongTensor([data[1] - 1 for data in data_list])

        # Create empty stores
        out = data_list[0][0].__class__()
        out.stores_as(data_list[0][0])

        # Group storage objects of every data object by key
        key_to_stores = defaultdict(list)
        for data in data_list:
            for store in data[0].stores:
                key_to_stores[store._key].append(store)

        # Iterate over each list of storage objects and recursively collate all its attributes.
        # `slice_dict` stores a compressed index representation of each attribute
        #  and is needed to re-construct individual elements from mini-batches.
        slice_dict = {}
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
                cat_dim = data_list[0][0].__cat_dim__(attr, values[0], stores[0])
                sizes = torch.tensor([value.size(cat_dim or 0) for value in values])
                slices = cumsum(sizes)
                value = torch.cat(values, dim=cat_dim or 0)

                out_store[attr] = value
                slice_dict[attr] = slices

        return out, targets, slice_dict
