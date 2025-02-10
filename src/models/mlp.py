import torch
from torch_geometric.data import Data
from torch_geometric.nn import MLP as PyGMLP  # noqa

from src.models.expansion import GaussianBasis


class MLP(PyGMLP):  # noqa
    def __init__(self, num_radial: int, *args, **kwargs) -> None:  # noqa
        super().__init__(*args, **kwargs)

        self.rbf = GaussianBasis(num_radial=num_radial)
        self.sbf = GaussianBasis(num_radial=num_radial)

        self.lin_edge = torch.nn.Linear(num_radial, self.out_channels, bias=False)
        self.lin_node = torch.nn.Linear(num_radial, self.out_channels)

    def forward(self, data: Data) -> torch.Tensor:  # noqa
        assert data.x is not None
        assert data.edge_attr is not None
        assert data.edge_index is not None
        assert data.num_nodes is not None

        x, edge_attr = data.x, data.edge_attr

        num_nodes, num_edges = data.num_nodes, edge_attr.size(0)
        num_radial = self.rbf.num_radial
        k = num_edges // num_nodes + 1
        num_atoms = num_nodes // k

        # Embed the features
        x: torch.Tensor = self.rbf(x)
        edge_attr: torch.Tensor = self.sbf(edge_attr)

        # Aggregate the features and reshape the tensor to the inverse line graph structure
        edge_attr = edge_attr.view(num_nodes, (k - 1) * num_radial)
        features = torch.cat([x, edge_attr], dim=1)
        features = features.view(num_atoms, k, -1).flatten(1)

        out = super().forward(features)

        return out


if __name__ == "__main__":
    import lightning as L
    import torchmetrics
    from lightning.pytorch.callbacks import RichModelSummary

    from src.datamodule import LightningDataset
    from src.module import Module
    from src.transforms import LineGraph, RandomPerturbation

    L.seed_everything(42)
    K = 10

    datamodule = LightningDataset(
        dataset_name="custom",
        lengths=(0.9, 0.1),
        use_imbalance_sampler=True,
        pre_transforms=LineGraph(),
        transforms=[
            RandomPerturbation(std=0.1),
        ],
        num_workers=5,
        k=K,
        rcut=6.0,
        batch_size=1,
        persistent_workers=True,
        # force_reload=True,
    )
    num_classes = datamodule.num_classes

    # Configure the Lightning Trainer
    trainer = L.Trainer(max_epochs=1, callbacks=[RichModelSummary()], deterministic=True)

    # Configure the Lightning Module
    with trainer.init_module():
        model = Module(
            model_name="mlp",
            num_classes=num_classes,
            compile=False,
            metrics=torchmetrics.MetricCollection(
                {
                    "acc": torchmetrics.Accuracy(
                        task="multiclass",
                        num_classes=num_classes,
                    ),
                    "f1": torchmetrics.F1Score(
                        task="multiclass",
                        num_classes=num_classes,
                    ),
                }
            ),
            warmup=100,
            max_iters=trainer.max_epochs * len(datamodule.train_dataloader()),  # type: ignore
            model_kwargs={
                "in_channels": -1,
                "num_layers": 2,
                "hidden_channels": 64,
                "num_radial": 25,
            },
        )

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
