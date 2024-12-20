import torch
from torch import Tensor
from torch.utils.data import WeightedRandomSampler
from torch_geometric.data import Data, Dataset, InMemoryDataset


class ImbalancedSampler(WeightedRandomSampler):
    """A weighted random sampler that randomly samples elements according to class distribution. As
    such, it will either remove samples from the majority class (under-sampling) or add more
    examples from the minority class (over-sampling).

    **Node-level sampling:**

    .. code-block:: python

        from torch_geometric.loader import NeighborLoader, ImbalancedSampler

        sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
        loader = NeighborLoader(data, input_nodes=data.train_mask,
                                batch_size=64, num_neighbors=[-1, -1],
                                sampler=sampler, ...)

    Args:
        dataset (Dataset or Data or Tensor): The dataset or class distribution
            from which to sample the data, given either as a
            `torch_geometric.data.Dataset`,
            `torch_geometric.data.Data`, or `torch.Tensor`
            object.
        input_nodes (Tensor, optional): The indices of nodes that are used by
            the corresponding loader, *e.g.*, by
            `torch_geometric.loader.NeighborLoader`.
            If set to `None`, all nodes will be considered.
            This argument should only be set for node-level loaders and does
            not have any effect when operating on a set of graphs as given by
            `torch_geometric.data.Dataset`. (default: `None`)
        num_samples (int, optional): The number of samples to draw for a single
            epoch. If set to `None`, will sample as much elements as there
            exists in the underlying data. (default: `None`)
    """

    def __init__(
        self,
        dataset: Dataset | Data | list[Data] | Tensor,
        input_nodes: Tensor | None = None,
        num_samples: int | None = None,
    ) -> None:
        if isinstance(dataset, Data):
            y = dataset.y.view(-1)  # type: ignore
            assert dataset.num_nodes == y.numel()
            y = y[input_nodes] if input_nodes is not None else y

        elif isinstance(dataset, Tensor):
            y = dataset.view(-1)
            y = y[input_nodes] if input_nodes is not None else y

        elif isinstance(dataset, InMemoryDataset):
            y = dataset.y.view(-1)
            assert len(dataset) == y.numel()

        else:
            ys = [data.y for data in dataset]
            if isinstance(ys[0], Tensor):
                y = torch.cat(ys, dim=0).view(-1)  # type: ignore
            else:
                y = torch.tensor(ys).view(-1)
            assert len(dataset) == y.numel()

        assert y.dtype == torch.long  # Require classification.

        num_samples = y.numel() if num_samples is None else num_samples

        class_weight = 1.0 / y.bincount()
        weight = class_weight[y]

        return super().__init__(weight, num_samples, replacement=True)  # type: ignore
