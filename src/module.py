from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torch_geometric.data import Data

from src.typing import SliceDictType


class CEGANNModule(LightningModule):
    """CEGANNModule is a PyTorch Lightning module that implements the CEGANN (Crystal Edge Graph
    Attention Neural Network) model.

    Args:
        model: The generator model.
        optimizer: The optimizer for training the model.
        scheduler: The learning rate scheduler.
        criterion: The loss criterion for training the model.
        metrics: Collection of metrics to evaluate the model performance.
        scheduler_params: Parameters for the scheduler.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        compile: bool,
        optimizer: Callable | torch.optim.Optimizer,
        scheduler: Callable | torch.optim.lr_scheduler._LRScheduler,
        metrics: torchmetrics.MetricCollection,
        scheduler_params: dict | None = None,
    ) -> None:
        super().__init__()

        if scheduler_params is None:
            scheduler_params = dict()
        self.scheduler_params = scheduler_params

        self.save_hyperparameters(logger=False, ignore=["model", "criterion", "metrics"])

        self.model = model
        if compile:
            self.model = torch.compile(self.model, fullgraph=False)

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x: Data) -> torch.Tensor:
        """Forward pass of the CEGANNModule.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, data: Data, batch_idx: int) -> torch.Tensor:
        """Training step of the CEGANNModule.

        Args:
            data: Input batch.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value.
        """
        preds: torch.Tensor = self(data)
        loss = F.cross_entropy(preds, torch.as_tensor(data.y, device=self.device))

        batch_value = self.train_metrics(preds.argmax(dim=-1), data.y)
        self.log_dict(batch_value, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Call hook method at the end of each training epoch."""
        self.train_metrics.reset()

    def validation_step(self, data: Data, batch_idx: int) -> None:
        """Validation step of the CEGANNModule.

        Args:
            data: Input batch.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value.
        """
        preds = self(data)
        self.val_metrics.update(preds.argmax(dim=-1), data.y)

    def on_validation_epoch_end(self) -> None:
        """Call hook method at the end of each validation epoch."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, data: Data, batch_idx: int) -> None:
        """Test step of the CEGANNModule.

        Args:
            data: Input batch.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value.
        """
        preds = self(data)
        self.test_metrics.update(preds, data.y)

    def on_test_epoch_end(self) -> None:
        """Call hook method at the end of each testing epoch."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    # TODO everything below needs to be updated
    def predict_step(
        self, batch: tuple[Data, torch.Tensor, SliceDictType], batch_idx: int
    ) -> torch.Tensor:
        """Predict step of the CEGANNModule.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Input batch.
            batch_idx (int): Index of the current batch.
        """
        preds = self.model_inference(batch)
        return preds

    def model_inference(self, batch: tuple[Data, torch.Tensor, SliceDictType]) -> torch.Tensor:
        """Perform an inference step of the model.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Input batch.

        Returns:
            torch.Tensor: Predicted labels.
        """
        x, _, _ = batch
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            Dictionary containing the optimizer and learning rate scheduler.
        """
        if isinstance(self.optimizer, Callable):
            optimizer = self.optimizer(params=self.parameters())

        if self.scheduler is not None:
            if isinstance(self.scheduler, Callable):
                scheduler = self.scheduler(
                    optimizer=optimizer,
                    **self.scheduler_params,
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
