import pytest
from src.datamodule import LightningDataset
from src.datasets import CustomDataset
from torch_geometric.loader import ImbalancedSampler


def test_datamodule_with_dataset_name():
    datamodule = LightningDataset(
        dataset_name="custom",
    )

    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is None
    assert datamodule.test_dataset is None

    datamodule.setup(stage="validate")
    with pytest.raises(AssertionError):
        datamodule.val_dataloader()

    datamodule.setup(stage="test")
    with pytest.raises(AssertionError):
        datamodule.test_dataloader()

    datamodule.setup(stage="predict")
    with pytest.raises(AssertionError):
        datamodule.predict_dataloader()


def test_datamodule_with_dataset():
    datamodule = LightningDataset(
        dataset=CustomDataset(),
    )

    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is None
    assert datamodule.test_dataset is None

    datamodule.setup(stage="validate")
    with pytest.raises(AssertionError):
        datamodule.val_dataloader()

    datamodule.setup(stage="test")
    with pytest.raises(AssertionError):
        datamodule.test_dataloader()

    datamodule.setup(stage="predict")
    with pytest.raises(AssertionError):
        datamodule.predict_dataloader()


def test_datamodule_with_lengths():
    datamodule = LightningDataset(
        dataset_name="custom",
        lengths=[0.8, 0.2],
    )

    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is None

    datamodule.setup(stage="validate")
    assert datamodule.val_dataloader() is not None

    datamodule.setup(stage="test")
    with pytest.raises(AssertionError):
        datamodule.test_dataloader()

    datamodule.setup(stage="predict")
    with pytest.raises(AssertionError):
        datamodule.predict_dataloader()


def test_datamodule_with_all_lengths():
    datamodule = LightningDataset(
        dataset_name="custom",
        lengths=[0.7, 0.2, 0.1],
    )

    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

    datamodule.setup(stage="validate")
    assert datamodule.val_dataloader() is not None

    datamodule.setup(stage="test")
    assert datamodule.test_dataloader() is not None

    datamodule.setup(stage="predict")
    with pytest.raises(AssertionError):
        datamodule.predict_dataloader()


def test_datamodule_with_pred_dataset():
    pred_dataset = CustomDataset()
    datamodule = LightningDataset(
        dataset_name="custom",
        pred_dataset=pred_dataset,
    )

    datamodule.setup(stage="predict")
    assert datamodule.pred_dataset is not None
    assert datamodule.predict_dataloader() is not None


def test_datamodule_with_imbalance_sampler():
    datamodule = LightningDataset(
        dataset_name="custom",
        use_imbalance_sampler=True,
    )

    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None

    train_dataloader = datamodule.train_dataloader()
    assert train_dataloader.sampler is not None
    assert isinstance(train_dataloader.sampler, ImbalancedSampler)


def test_lightning_dataset_initialization():
    dataset = CustomDataset()
    datamodule = LightningDataset(dataset=dataset, batch_size=2)
    assert datamodule.batch_size == 2
    assert datamodule.dataset == dataset


def test_lightning_dataset_invalid_dataset_name():
    with pytest.raises(ValueError):
        LightningDataset(dataset_name="invalid")


def test_lightning_dataset_no_dataset_or_name():
    with pytest.raises(ValueError):
        LightningDataset()


def test_lightning_dataset_invalid_lengths():
    dataset = CustomDataset()
    with pytest.raises(ValueError):
        LightningDataset(dataset=dataset, lengths=[1, 2, 3, 4])


def test_lightning_dataset_sampler_conflict():
    dataset = CustomDataset()
    with pytest.raises(ValueError):
        LightningDataset(dataset=dataset, use_imbalance_sampler=True, sampler="some_sampler")
