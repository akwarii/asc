import torch
import torchmetrics
from lightning import LightningModule


class CEGANNModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: torch.nn.Module,
        metrics: torchmetrics.MetricCollection,
        compile: bool = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            logger=False, ignore=["model", "criterion", "metrics"]
        )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = criterion

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.val_best_acc = torchmetrics.MaxMetric(prefix="val/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()
        self.val_best_acc.reset()

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self.model_step(batch)

        output = self.train_metrics(preds, targets)
        self.log_dict(output, on_step=True, on_epoch=False, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_metrics.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        output = self.val_metrics.compute()
        self.val_best_acc(output["val/acc"])

        self.log("val/acc_best", self.val_best_acc.compute(), sync_dist=True, prog_bar=True)
        self.log_dict(output)
        self.val_metrics.reset()

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_metrics.update(preds, targets)

    def on_test_epoch_end(self) -> None:
        output = self.test_metrics.compute()
        self.log_dict(output)
        self.test_metrics.reset()
        
    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> dict[str, torch.optim.lr_scheduler._LRScheduler]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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
