from typing import Any

import torch
import torch.nn.functional as F
import torchmetrics
from lightning import LightningModule
from torch_geometric.data import Data

from src import models
from src.optim import get_cosine_schedule_with_warmup

MODEL_FACTORY = {
    "cegann": models.CEGANN,
    "mlp": models.MLP,
    "gat": models.GATClassifier,
}


class Module(LightningModule):
    """PyTorch Lightning module used for training and evaluating multi-class classification models.

    Args:
        model_name (str): The name of the model.
        optimizer (Callable | torch.optim.Optimizer): The optimizer for training the model.
        metrics (torchmetrics.MetricCollection): Collection of metrics to evaluate the model
            performance.
        compile (bool, optional): Whether to compile the model. Defaults to True.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
        scheduler (Callable | torch.optim.lr_scheduler._LRScheduler, optional): The learning rate
            scheduler. Defaults to None.
        scheduler_params (dict[str, Any], optional): Parameters for the scheduler. Defaults to
            None.
        model_kwargs (dict[str, Any], optional): Additional keyword arguments for the model.
            Defaults to None.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        metrics: torchmetrics.MetricCollection,
        compile: bool = True,
        lr: float = 1e-3,
        warmup: int = 100,
        max_iters: int = 1_000,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["metrics"])
        self._create_model()

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def _create_model(self) -> None:
        """Create the model using its name and kwargs given to the module construstor."""
        model_name = self.hparams["model_name"].lower()
        model_kwargs = self.hparams["model_kwargs"]
        out_channels = self.hparams["num_classes"]

        if model_kwargs is None:
            model_kwargs = dict()

        model = MODEL_FACTORY.get(model_name, None)
        if model is None:
            raise NotImplementedError(
                f"Model {model_name} is not implemented. Available models: {MODEL_FACTORY.keys()}"
            )

        self.model = model(out_channels=out_channels, **model_kwargs)

        if self.hparams["compile"]:
            self.model = torch.compile(self.model)

    def forward(self, x: Data) -> torch.Tensor:
        """Forward pass of the CEGANNModule.

        Args:
            x: Input data.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, data: Data) -> torch.Tensor:
        """Training step of the CEGANNModule.

        Args:
            data: Input batch.
            batch_idx: Index of the current batch.

        Returns:
            torch.Tensor: Loss value.
        """
        preds: torch.Tensor = self(data)
        loss = F.cross_entropy(preds, torch.as_tensor(data.y))

        batch_value = self.train_metrics(preds.softmax(dim=-1), data.y)
        self.log_dict(
            batch_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=data.num_nodes,
        )

        return loss

    def on_train_epoch_start(self) -> None:
        """Call hook method at the start of each training epoch."""
        if self.trainer.sanity_checking:
            return

        if self.trainer.progress_bar_callback is None:
            print(f"Epoch {self.trainer.current_epoch} started")

    def on_train_epoch_end(self) -> None:
        """Call hook method at the end of each training epoch."""
        self.train_metrics.reset()

    def validation_step(self, data: Data) -> None:
        """Validation step of the CEGANNModule.

        Args:
            data: Input batch.
        """
        preds: torch.Tensor = self(data)
        self.val_metrics.update(preds.softmax(dim=-1), data.y)

    def on_validation_epoch_end(self) -> None:
        """Call hook method at the end of each validation epoch."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, data: Data) -> None:
        """Test step of the CEGANNModule.

        Args:
            data: Input batch.
        """
        preds: torch.Tensor = self(data)
        self.test_metrics.update(preds.softmax(dim=-1), data.y)

    def on_test_epoch_end(self) -> None:
        """Call hook method at the end of each testing epoch."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, data: Data) -> torch.Tensor:
        """Predict step of the CEGANNModule.

        Args:
            data: Input batch.

        Returns:
            torch.Tensor: Predicted class labels.
        """
        return torch.argmax(self(data), dim=-1)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.hparams["warmup"],
            num_training_steps=self.hparams["max_iters"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "monitor": "val/loss",
                "frequency": 1,
            },
        }
