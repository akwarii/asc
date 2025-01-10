import torch
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
        compile: Whether to compile the model using torch.compile().
        scheduler_params: Parameters for the scheduler.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: torch.nn.Module,
        metrics: torchmetrics.MetricCollection,
        compile: bool = True,
        scheduler_params: dict = {},  # TODO never use a mutable default argument
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["model", "criterion", "metrics"])

        # TODO add model compilation
        # DB : https://lightning.ai/docs/pytorch/latest/advanced/compile.html
        # DB : https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = criterion

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        # # self.val_best_acc = torchmetrics.MaxMetric(prefix="val/")
        self.val_best_acc = (
            torchmetrics.MaxMetric()
        )  # modified by DB, no prefix keyword for MaxMetric

    def forward(self, x: Data) -> torch.Tensor:
        """Forward pass of the CEGANNModule.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def on_train_start(self) -> None:
        """Call hook method when the training starts."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()
        self.val_best_acc.reset()

    def model_step(
        self, batch: tuple[Data, torch.Tensor, SliceDictType]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single step of the model.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Input batch.

        Returns:
            Tuple containing the loss, predicted labels, and target labels.
        """
        x, y, slices = batch

        #TODO when batch size is 1, slices is None and the following line will raise an error
        _y = torch.cat(
            [y[i].repeat(slices["pos"][i + 1] - slices["pos"][i]) for i in range(y.size()[0])]
        )  # TODO can it be simplified?

        logits = self.forward(x)

        loss = self.criterion(logits, _y)  # DB
        preds = torch.argmax(logits, dim=1)
        return loss, preds, _y

    def training_step(
        self, batch: tuple[Data, torch.Tensor, SliceDictType], batch_idx: int
    ) -> torch.Tensor:
        """Training step of the CEGANNModule.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Input batch.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value.
        """
        loss, preds, targets = self.model_step(batch)

        output = self.train_metrics(preds, targets)
        self.log_dict(output, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Call hook method at the end of each training epoch."""
        print("")  # DB, to avoid overlap between progress bars
        # pass

    def validation_step(
        self, batch: tuple[Data, torch.Tensor, SliceDictType], batch_idx: int
    ) -> None:
        """Perform validation step of the CEGANNModule.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Input batch.
            batch_idx (int): Index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.val_metrics.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        """Call hook method at the end of each validation epoch."""
        output = self.val_metrics.compute()
        self.val_best_acc(output["val/accuracy"])

        self.log("val/acc_best", self.val_best_acc.compute(), sync_dist=True, prog_bar=True)
        self.log_dict(output)
        self.val_metrics.reset()

    def test_step(self, batch: tuple[Data, torch.Tensor, SliceDictType], batch_idx: int) -> None:
        """Test step of the CEGANNModule.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Input batch.
            batch_idx (int): Index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)
        self.test_metrics.update(preds, targets)

    def on_test_epoch_end(self) -> None:
        """Call hook method at the end of each testing epoch."""
        output = self.test_metrics.compute()
        self.log_dict(output)
        self.test_metrics.reset()

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

    # DB
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

    def setup(self, stage: str) -> None:
        """Set up the model before training/evaluation.

        Args:
            stage (str): Either "fit" or "test".
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> dict[str, torch.optim.lr_scheduler._LRScheduler]:
        """Configure the optimizer and learning rate scheduler.

        Returns:
            Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer=optimizer,
                **self.hparams.scheduler_params,  # DB
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
