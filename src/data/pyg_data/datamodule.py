import copy
import inspect
import warnings
from typing import Any, Type

from lightning import LightningDataModule as PLLightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler import BaseSampler, NeighborSampler
from torch_geometric.typing import InputNodes, OptTensor


class LightningDataModule(PLLightningDataModule):
    def __init__(self, has_val: bool, has_test: bool, **kwargs: Any) -> None:
        super().__init__()

        if not has_val:
            self.val_dataloader = None  # type: ignore

        if not has_test:
            self.test_dataloader = None  # type: ignore

        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('num_workers', 0)
        kwargs.setdefault('pin_memory', True)
        kwargs.setdefault('persistent_workers',
                          kwargs.get('num_workers', 0) > 0)

        if 'shuffle' in kwargs:
            warnings.warn(f"The 'shuffle={kwargs['shuffle']}' option is "
                          f"ignored in '{self.__class__.__name__}'. Remove it "
                          f"from the argument list to disable this warning")
            del kwargs['shuffle']

        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({kwargs_repr(**self.kwargs)})'


class LightningData(LightningDataModule):
    def __init__(
        self,
        data: Data,
        has_val: bool,
        has_test: bool,
        loader: str = 'neighbor',
        graph_sampler: BaseSampler | None = None,
        eval_loader_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault('batch_size', 1)
        kwargs.setdefault('num_workers', 0)

        if graph_sampler is not None:
            loader = 'custom'

        # For full-batch training, we use reasonable defaults for a lot of
        # data-loading options:
        if loader not in ['neighbor', 'custom']:
            raise ValueError(f"Undefined 'loader' option (got '{loader}')")

        super().__init__(has_val, has_test, **kwargs)

        self.data = data
        self.loader = loader

        # Determine sampler and loader arguments ##############################

        if loader == 'neighbor':

            # Define a new `NeighborSampler` to be re-used across data loaders:
            sampler_kwargs, self.loader_kwargs = split_kwargs(
                self.kwargs,
                NeighborSampler,
            )
            sampler_kwargs.setdefault('share_memory', self.kwargs['num_workers'] > 0)
            self.graph_sampler: BaseSampler = NeighborSampler(
                data, **sampler_kwargs)

        elif graph_sampler is not None: # custom loader
            sampler_kwargs, self.loader_kwargs = split_kwargs(
                self.kwargs,
                graph_sampler.__class__,
            )
            if len(sampler_kwargs) > 0:
                warnings.warn(f"Ignoring the arguments "
                              f"{list(sampler_kwargs.keys())} in "
                              f"'{self.__class__.__name__}' since a custom "
                              f"'graph_sampler' was passed")
            self.graph_sampler = graph_sampler

        # Determine validation sampler and loader arguments ###################

        self.eval_loader_kwargs = copy.copy(self.loader_kwargs)
        if eval_loader_kwargs is not None:
            # If the user wants to override certain values during evaluation,
            # we shallow-copy the graph sampler and update its attributes.
            if hasattr(self, 'graph_sampler'):
                self.eval_graph_sampler = copy.copy(self.graph_sampler)

                eval_sampler_kwargs, eval_loader_kwargs = split_kwargs(
                    eval_loader_kwargs,
                    self.graph_sampler.__class__,
                )
                for key, value in eval_sampler_kwargs.items():
                    setattr(self.eval_graph_sampler, key, value)

            self.eval_loader_kwargs.update(eval_loader_kwargs)

        elif hasattr(self, 'graph_sampler'):
            self.eval_graph_sampler = self.graph_sampler

        self.eval_loader_kwargs.pop('sampler', None)
        self.eval_loader_kwargs.pop('batch_sampler', None)

        if 'batch_sampler' in self.loader_kwargs:
            self.loader_kwargs.pop('batch_size', None)

    @property
    def train_shuffle(self) -> bool:
        shuffle = self.loader_kwargs.get('sampler', None) is None
        shuffle &= self.loader_kwargs.get('batch_sampler', None) is None
        return shuffle

    def prepare_data(self) -> None:
        super().prepare_data()

    def __repr__(self) -> str:
        kwargs = kwargs_repr(data=self.data, loader=self.loader, **self.kwargs)
        return f'{self.__class__.__name__}({kwargs})'


class LightningNodeData(LightningData):
    r"""Converts a :class:`~torch_geometric.data.Data` object into a
    :class:`pytorch_lightning.LightningDataModule` variant. It can then be
    automatically used as a :obj:`datamodule` for multi-GPU node-level
    training via :lightning:`null`
    `PyTorch Lightning <https://www.pytorchlightning.ai>`__.
    :class:`LightningDataset` will take care of providing mini-batches via
    :class:`~torch_geometric.loader.NeighborLoader`.

    Args:
        data (Data): The :class:`~torch_geometric.data.Data` graph object.
        input_train_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of training nodes.
            If not given, will try to automatically infer them from the
            :obj:`data` object by searching for :obj:`train_mask`,
            :obj:`train_idx`, or :obj:`train_index` attributes.
            (default: :obj:`None`)
        input_train_time (torch.Tensor, optional): The timestamp
            of training nodes. (default: :obj:`None`)
        input_val_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of validation nodes.
            If not given, will try to automatically infer them from the
            :obj:`data` object by searching for :obj:`val_mask`,
            :obj:`valid_mask`, :obj:`val_idx`, :obj:`valid_idx`,
            :obj:`val_index`, or :obj:`valid_index` attributes.
            (default: :obj:`None`)
        input_val_time (torch.Tensor, optional): The timestamp
            of validation edges. (default: :obj:`None`)
        input_test_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of test nodes.
            If not given, will try to automatically infer them from the
            :obj:`data` object by searching for :obj:`test_mask`,
            :obj:`test_idx`, or :obj:`test_index` attributes.
            (default: :obj:`None`)
        input_test_time (torch.Tensor, optional): The timestamp
            of test nodes. (default: :obj:`None`)
        input_pred_nodes (torch.Tensor or str or (str, torch.Tensor)): The
            indices of prediction nodes.
            If not given, will try to automatically infer them from the
            :obj:`data` object by searching for :obj:`pred_mask`,
            :obj:`pred_idx`, or :obj:`pred_index` attributes.
            (default: :obj:`None`)
        input_pred_time (torch.Tensor, optional): The timestamp
            of prediction nodes. (default: :obj:`None`)
        node_sampler (BaseSampler, optional): A custom sampler object to
            generate mini-batches. If set, will ignore the :obj:`loader`
            option. (default: :obj:`None`)
        eval_loader_kwargs (dict[str, Any], optional): Custom keyword arguments
            that override the :class:`torch_geometric.loader.NeighborLoader`
            configuration during evaluation. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.loader.NeighborLoader`.
    """
    def __init__(
        self,
        data: Data,
        input_train_nodes: InputNodes = None,
        input_train_time: OptTensor = None,
        input_val_nodes: InputNodes = None,
        input_val_time: OptTensor = None,
        input_test_nodes: InputNodes = None,
        input_test_time: OptTensor = None,
        input_pred_nodes: InputNodes = None,
        input_pred_time: OptTensor = None,
        node_sampler: BaseSampler | None = None,
        eval_loader_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if input_train_nodes is None:
            input_train_nodes = infer_input_nodes(data, split='train')

        if input_val_nodes is None:
            input_val_nodes = infer_input_nodes(data, split='val')
            if input_val_nodes is None:
                input_val_nodes = infer_input_nodes(data, split='valid')

        if input_test_nodes is None:
            input_test_nodes = infer_input_nodes(data, split='test')

        if input_pred_nodes is None:
            input_pred_nodes = infer_input_nodes(data, split='pred')

        super().__init__(
            data=data,
            has_val=input_val_nodes is not None,
            has_test=input_test_nodes is not None,
            loader="neighbor",
            graph_sampler=node_sampler,
            eval_loader_kwargs=eval_loader_kwargs,
            **kwargs,
        )

        self.input_train_nodes = input_train_nodes
        self.input_train_time = input_train_time
        self.input_train_id: OptTensor = None

        self.input_val_nodes = input_val_nodes
        self.input_val_time = input_val_time
        self.input_val_id: OptTensor = None

        self.input_test_nodes = input_test_nodes
        self.input_test_time = input_test_time
        self.input_test_id: OptTensor = None

        self.input_pred_nodes = input_pred_nodes
        self.input_pred_time = input_pred_time
        self.input_pred_id: OptTensor = None

    def dataloader(
        self,
        input_nodes: InputNodes,
        input_time: OptTensor = None,
        input_id: OptTensor = None,
        node_sampler: BaseSampler | None = None,
        **kwargs: Any,
    ) -> DataLoader:

        assert node_sampler is not None

        return NodeLoader(
            self.data,
            node_sampler=node_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            input_id=input_id,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.input_train_nodes,
            self.input_train_time,
            self.input_train_id,
            node_sampler=getattr(self, 'graph_sampler', None),
            shuffle=self.train_shuffle,
            **self.loader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.input_val_nodes,
            self.input_val_time,
            self.input_val_id,
            node_sampler=getattr(self, 'eval_graph_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.input_test_nodes,
            self.input_test_time,
            self.input_test_id,
            node_sampler=getattr(self, 'eval_graph_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.input_pred_nodes,
            self.input_pred_time,
            self.input_pred_id,
            node_sampler=getattr(self, 'eval_graph_sampler', None),
            shuffle=False,
            **self.eval_loader_kwargs,
        )
        

def infer_input_nodes(data: Data, split: str) -> InputNodes:
    attr_name: str | None = None
    if f'{split}_mask' in data:
        attr_name = f'{split}_mask'
    elif f'{split}_idx' in data:
        attr_name = f'{split}_idx'
    elif f'{split}_index' in data:
        attr_name = f'{split}_index'

    if attr_name is None:
        return None

    if isinstance(data, Data):
        return data[attr_name]


def kwargs_repr(**kwargs: Any) -> str:
    return ', '.join([f'{k}={v}' for k, v in kwargs.items() if v is not None])


def split_kwargs(
    kwargs: dict[str, Any],
    sampler_cls: Type,
) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""Splits keyword arguments into sampler and loader arguments."""
    sampler_args = inspect.signature(sampler_cls).parameters

    sampler_kwargs: dict[str, Any] = {}
    loader_kwargs: dict[str, Any] = {}

    for key, value in kwargs.items():
        if key in sampler_args:
            sampler_kwargs[key] = value
        else:
            loader_kwargs[key] = value

    return sampler_kwargs, loader_kwargs
