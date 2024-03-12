from collections.abc import Generator, Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data

from src.utils.typing import PathLike


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

    def _get_graph_data(self, struct: Structure):
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

        return edge_index, edge_distances

    @staticmethod
    def _to_pymatgen_struct(struct_repr: str | Path) -> Structure:
        if isinstance(struct_repr, str):
            struct = Structure.from_str(struct_repr, fmt="poscar")
        elif isinstance(struct, Path):
            if not struct_repr.is_file():
                raise FileNotFoundError(f"The file {struct_repr} does not exist.")
            struct = Structure.from_file(struct_repr)
        elif isinstance(struct_repr, Structure):
            pass
        else:
            raise ValueError("The input must be a pymatgen structure object, a string or a path.")

        return struct

    def convert(
        self,
        struct: Structure,
        mask_sites: torch.BoolTensor = None,
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
        struct = self._to_pymatgen_struct(struct)

        edge_index, edge_distances = self._get_graph_data(struct)
        num_nodes = len(struct)

        if mask_sites is not None:
            assert (
                len(mask_sites.dim()) == 1 and len(mask_sites) == num_nodes
            ), "The number of tags must match the number of atoms in the structure."
            assert mask_sites.dtype == torch.bool, "The mask must be a boolean tensor."

        # put the minimum data in torch geometric data object
        data = Data(
            num_nodes=num_nodes,
            pos=torch.FloatTensor(struct.cart_coords),
            cell=torch.FloatTensor(struct.lattice.matrix.copy()),
            edge_index=edge_index,
            edge_dist=edge_distances,
            mask=mask_sites,
        )

        return data

    def batch_conversion(
        self,
        structs: Iterable[Structure | str | Path],
        mask_struct_sites: dict[int, torch.BoolTensor] = None,
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
        structs: Iterable[Structure | str | Path],
        path: PathLike,
        mask_struct_sites: dict[int, torch.BoolTensor] = None,
        progress_bar: bool = True,
    ) -> None:
        """Convert a list of atomic structures to a list of graphs and save them to a file. Note
        that the whole list of graphs must fit in memory.

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
        graphs = list(self.batch_conversion(structs, mask_struct_sites, progress_bar))

        torch.save(graphs, path)

        return None

    @staticmethod
    def _graph_collate(data_list: Sequence[Data]) -> ...:
        if len(data_list) == 0:
            raise ValueError("The list of graphs is empty.")
        elif len(data_list) == 1:
            return data_list[0], None


from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch_geometric.typing
from torch import Tensor
from torch_geometric import EdgeIndex
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage, NodeStorage
from torch_geometric.edge_index import SortOrder
from torch_geometric.typing import SparseTensor, TensorFrame, torch_frame, torch_sparse
from torch_geometric.utils import cumsum, is_sparse, is_torch_sparse_tensor
from torch_geometric.utils.sparse import cat

T = TypeVar("T")
SliceDictType = dict[str, Union[Tensor, dict[str, Tensor]]]
IncDictType = dict[str, Union[Tensor, dict[str, Tensor]]]


def collate(
    cls: type[T],
    data_list: list[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Iterable[str] | None = None,
    exclude_keys: Iterable[str] | None = None,
) -> tuple[T, SliceDictType, IncDictType]:
    # Collates a list of `data` objects into a single object of type `cls`.
    # `collate` can handle both homogeneous and heterogeneous data objects by
    # individually collating all their stores.
    # In addition, `collate` can handle nested data structures such as
    # dictionaries and lists.

    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:  # Dynamic inheritance.
        out = cls(_base_cls=data_list[0].__class__)  # type: ignore
    else:
        out = cls()

    # Create empty stores:
    out.stores_as(data_list[0])  # type: ignore

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_stores = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    device: torch.device | None = None
    slice_dict: SliceDictType = {}
    inc_dict: IncDictType = {}
    for out_store in out.stores:  # type: ignore
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():

            if attr in exclude_keys:  # Do not include top-level attribute.
                continue

            values = [store[attr] for store in stores]

            # The `num_nodes` attribute needs special treatment, as we need to
            # sum their values up instead of merging them to a list:
            if attr == "num_nodes":
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            # Skip batching of `ptr` vectors for now:
            if attr == "ptr":
                continue

            # Collate attributes into a unified representation:
            value, slices, incs = _collate(attr, values, data_list, stores, increment)

            # If parts of the data are already on GPU, make sure that auxiliary
            # data like `batch` or `ptr` are also created on GPU:
            if isinstance(value, Tensor) and value.is_cuda:
                device = value.device

            out_store[attr] = value

            if key is not None:  # Heterogeneous:
                store_slice_dict = slice_dict.get(key, {})
                assert isinstance(store_slice_dict, dict)
                store_slice_dict[attr] = slices
                slice_dict[key] = store_slice_dict

                store_inc_dict = inc_dict.get(key, {})
                assert isinstance(store_inc_dict, dict)
                store_inc_dict[attr] = incs
                inc_dict[key] = store_inc_dict
            else:  # Homogeneous:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute:
            if attr in follow_batch:
                batch, ptr = _batch_and_ptr(slices, device)
                out_store[f"{attr}_batch"] = batch
                out_store[f"{attr}_ptr"] = ptr

        # In case of node-level storages, we add a top-level batch vector it:
        if add_batch and isinstance(stores[0], NodeStorage) and stores[0].can_infer_num_nodes:
            repeats = [store.num_nodes or 0 for store in stores]
            out_store.batch = repeat_interleave(repeats, device=device)
            out_store.ptr = cumsum(torch.tensor(repeats, device=device))

    return out, slice_dict, inc_dict


def _collate(
    key: str,
    values: list[Any],
    data_list: list[BaseData],
    stores: list[BaseStorage],
    increment: bool,
) -> tuple[Any, Any, Any]:

    elem = values[0]

    if isinstance(elem, Tensor) and not is_sparse(elem):
        # Concatenate a list of `torch.Tensor` along the `cat_dim`.
        # NOTE: We need to take care of incrementing elements appropriately.
        key = str(key)
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        if cat_dim is None or elem.dim() == 0:
            values = [value.unsqueeze(0) for value in values]
        sizes = torch.tensor([value.size(cat_dim or 0) for value in values])
        slices = cumsum(sizes)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if incs.dim() > 1 or int(incs[-1]) != 0:
                values = [value + inc.to(value.device) for value, inc in zip(values, incs)]
        else:
            incs = None

        if getattr(elem, "is_nested", False):
            tensors = []
            for nested_tensor in values:
                tensors.extend(nested_tensor.unbind())
            value = torch.nested.nested_tensor(tensors)

            return value, slices, incs

        out = None
        if torch.utils.data.get_worker_info() is not None:
            # Write directly into shared memory to avoid an extra copy:
            numel = sum(value.numel() for value in values)
            if torch_geometric.typing.WITH_PT20:
                storage = elem.untyped_storage()._new_shared(
                    numel * elem.element_size(), device=elem.device
                )
            elif torch_geometric.typing.WITH_PT112:
                storage = elem.storage()._new_shared(numel, device=elem.device)
            else:
                storage = elem.storage()._new_shared(numel)
            shape = list(elem.size())
            if cat_dim is None or elem.dim() == 0:
                shape = [len(values)] + shape
            else:
                shape[cat_dim] = int(slices[-1])
            out = elem.new(storage).resize_(*shape)

        value = torch.cat(values, dim=cat_dim or 0, out=out)

        if increment and isinstance(value, EdgeIndex) and values[0].is_sorted:
            # Check whether the whole `EdgeIndex` is sorted by row:
            if values[0].is_sorted_by_row and (value[0].diff() >= 0).all():
                value._sort_order = SortOrder.ROW
            # Check whether the whole `EdgeIndex` is sorted by column:
            elif values[0].is_sorted_by_col and (value[1].diff() >= 0).all():
                value._sort_order = SortOrder.COL

        return value, slices, incs

    elif isinstance(elem, TensorFrame):
        key = str(key)
        sizes = torch.tensor([value.num_rows for value in values])
        slices = cumsum(sizes)
        value = torch_frame.cat(values, dim=0)
        return value, slices, None

    elif is_sparse(elem) and increment:
        # Concatenate a list of `SparseTensor` along the `cat_dim`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        key = str(key)
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        cat_dims = (cat_dim,) if isinstance(cat_dim, int) else cat_dim
        repeats = [[value.size(dim) for dim in cat_dims] for value in values]
        slices = cumsum(torch.tensor(repeats))
        if is_torch_sparse_tensor(elem):
            value = cat(values, dim=cat_dim)
        else:
            value = torch_sparse.cat(values, dim=cat_dim)
        return value, slices, None

    elif isinstance(elem, (int, float)):
        # Convert a list of numerical values to a `torch.Tensor`.
        value = torch.tensor(values)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if int(incs[-1]) != 0:
                value.add_(incs)
        else:
            incs = None
        slices = torch.arange(len(values) + 1)
        return value, slices, incs

    elif isinstance(elem, Mapping):
        # Recursively collate elements of dictionaries.
        value_dict, slice_dict, inc_dict = {}, {}, {}
        for key in elem.keys():
            value_dict[key], slice_dict[key], inc_dict[key] = _collate(
                key, [v[key] for v in values], data_list, stores, increment
            )
        return value_dict, slice_dict, inc_dict

    elif (
        isinstance(elem, Sequence)
        and not isinstance(elem, str)
        and len(elem) > 0
        and isinstance(elem[0], (Tensor, SparseTensor))
    ):
        # Recursively collate elements of lists.
        value_list, slice_list, inc_list = [], [], []
        for i in range(len(elem)):
            value, slices, incs = _collate(
                key, [v[i] for v in values], data_list, stores, increment
            )
            value_list.append(value)
            slice_list.append(slices)
            inc_list.append(incs)
        return value_list, slice_list, inc_list

    else:
        # Other-wise, just return the list of values as it is.
        slices = torch.arange(len(values) + 1)
        return values, slices, None


def _batch_and_ptr(
    slices: Any,
    device: torch.device | None = None,
) -> tuple[Any, Any]:
    if isinstance(slices, Tensor) and slices.dim() == 1:
        # Default case, turn slices tensor into batch.
        repeats = slices[1:] - slices[:-1]
        batch = repeat_interleave(repeats.tolist(), device=device)
        ptr = cumsum(repeats.to(device))
        return batch, ptr

    elif isinstance(slices, Mapping):
        # Recursively batch elements of dictionaries.
        batch, ptr = {}, {}
        for k, v in slices.items():
            batch[k], ptr[k] = _batch_and_ptr(v, device)
        return batch, ptr

    elif (
        isinstance(slices, Sequence)
        and not isinstance(slices, str)
        and isinstance(slices[0], Tensor)
    ):
        # Recursively batch elements of lists.
        batch, ptr = [], []
        for s in slices:
            sub_batch, sub_ptr = _batch_and_ptr(s, device)
            batch.append(sub_batch)
            ptr.append(sub_ptr)
        return batch, ptr

    else:
        # Failure of batching, usually due to slices.dim() != 1
        return None, None


def repeat_interleave(
    repeats: list[int],
    device: torch.device | None = None,
) -> Tensor:
    outs = [torch.full((n,), i, device=device) for i, n in enumerate(repeats)]
    return torch.cat(outs, dim=0)


def get_incs(
    key, values: list[Any], data_list: list[BaseData], stores: list[BaseStorage]
) -> Tensor:
    repeats = [
        data.__inc__(key, value, store) for value, data, store in zip(values, data_list, stores)
    ]
    if isinstance(repeats[0], Tensor):
        repeats = torch.stack(repeats, dim=0)
    else:
        repeats = torch.tensor(repeats)
    return cumsum(repeats[:-1])


if __name__ == "__main__":
    import pandas as pd

    knn = KNNGraph(k=20, rcut=10)

    input_dir = Path("data") / "csg" / "raw"

    df = pd.read_csv(input_dir / "CSG.csv")
    struct_list = df["Structure"].to_list()

    for graph in knn.batch_conversion(struct_list, progress_bar=True):
        ...
