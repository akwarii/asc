from torch_geometric.data.lightning import LightningDataset


class CEGANNLightningDataset(LightningDataset):
    """A wrapper around LightningDataset that sets the batch size as an attribute after
    initialization. It is only used to have a direct access to the batch size in the datamodule,
    which is expected by the BatchSizeFinder callback of Lightning.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset, optional): The validation dataset.
            (default: :obj:`None`)
        test_dataset (Dataset, optional): The test dataset.
            (default: :obj:`None`)
        pred_dataset (Dataset, optional): The prediction dataset.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.loader.DataLoader`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_size = self.kwargs.get("batch_size", 1)
        self.kwargs["batch_size"] = self._batch_size

    @property
    def batch_size(self) -> int:
        """The batch size to be used in the dataloader."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value) -> None:
        self._batch_size = value
        self.kwargs["batch_size"] = value
