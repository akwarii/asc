from typing import Any

import torch
import torch.nn as nn
from lightning import LightningModule
from packaging.version import Version
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.utilities.types import LRSchedulerType
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Data
from torchmetrics import MetricCollection

from src import models
from src.optim import get_cosine_schedule_with_warmup

MODEL_FACTORY: dict[str, nn.Module] = {
    "cegann": models.CEGANN,
    "mlp": models.MLPClassifier,
    "gat": models.GATClassifier,
    "cegannv2": models.CEGANNv2,
    "painn": models.PaiNN,
}  # type: ignore


class Module(LightningModule):
    """PyTorch Lightning module used for training and evaluating multi-class classification models.

    Args:
        model_name (str): The name of the model.
        num_classes (int): The number of classes in the dataset.
        metrics (torchmetrics.MetricCollection): Collection of metrics to evaluate the model
            performance.
        compile (bool, optional): Whether to compile the model. Defaults to True.
        lr (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
        warmup (int, optional): The number of warmup steps for the learning rate scheduler.
            Defaults to 100.
        max_iters (int, optional): The maximum number of iterations for the learning rate
            scheduler. Defaults to 1_000.
        model_kwargs (dict[str, Any], optional): Additional keyword arguments for the model.
            Defaults to None.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        *,
        metrics: MetricCollection | None = None,
        compile: bool = True,
        lr: float = 1e-3,
        warmup: int = 100,
        max_iters: int = 1_000,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.automatic_optimization = False

        self.can_compile = True
        if torch.cuda.is_available() and (
            torch.cuda.get_device_capability() < (7, 0)
            or Version(torch.__version__) < Version("2.0")
        ):
            print(
                "Warning: torch.compile is not supported on this device or PyTorch version. "
                "Proceeding without compilation."
            )
            self.can_compile = False

        self.save_hyperparameters(logger=False, ignore=["metrics"])
        self._create_model()

        self.criterion = nn.CrossEntropyLoss()

        if metrics is not None:
            self.train_metrics = metrics.clone(prefix="train/")
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")

    def _create_model(self) -> None:
        model_name = self.hparams["model_name"].lower()
        model_kwargs = self.hparams["model_kwargs"]
        out_channels = self.hparams["num_classes"]

        if model_kwargs is None:
            model_kwargs = dict()

        if model_name == "mlp":
            model_kwargs["in_channels"] = model_kwargs["k"] ** 2 * model_kwargs["num_radial"]

        model_kwargs.pop("k", None)

        model = MODEL_FACTORY.get(model_name, None)
        if model is None:
            raise NotImplementedError(
                f"Model {model_name} is not implemented. Available models: {MODEL_FACTORY.keys()}"
            )

        self.model = model(out_channels=out_channels, **model_kwargs)

        if self.hparams["compile"] and self.can_compile:
            self.model = torch.compile(self.model, fullgraph=True)

    def forward(self, data: Data) -> torch.Tensor:
        return self.model(data)

    def training_step(self, data: Data) -> torch.Tensor:
        opts = self.optimizers()
        schs = self.lr_schedulers()

        if not isinstance(opts, list):
            opts = [opts]
        if schs is not None and not isinstance(schs, list):
            schs = [schs]

        preds: torch.Tensor = self(data)
        loss = self.criterion(preds, torch.as_tensor(data.y))

        for opt in opts:
            opt.zero_grad(set_to_none=True)

        self.manual_backward(loss)

        self._optimization_step(opts, schs)

        if hasattr(self, "train_metrics"):
            batch_value = self.train_metrics(preds.softmax(dim=-1), data.y)
            self.log_dict(
                batch_value,
                prog_bar=True,
                batch_size=data.num_nodes,
            )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            batch_size=data.num_nodes,
        )

        return loss

    def on_train_epoch_start(self) -> None:
        if self.trainer.sanity_checking:
            return

        if self.trainer.progress_bar_callback is None:
            print(f"Epoch {self.trainer.current_epoch} started")

    def on_train_epoch_end(self) -> None:
        if hasattr(self, "train_metrics"):
            self.train_metrics.reset()

    def validation_step(self, data: Data) -> None:
        preds: torch.Tensor = self(data)
        loss = self.criterion(preds, torch.as_tensor(data.y))

        if hasattr(self, "val_metrics"):
            self.val_metrics.update(preds.softmax(dim=-1), data.y)
        self.log("val/loss", loss, on_epoch=True, batch_size=data.num_nodes)

    def on_validation_epoch_end(self) -> None:
        if hasattr(self, "val_metrics"):
            self.log_dict(self.val_metrics.compute())
            self.val_metrics.reset()

    def test_step(self, data: Data) -> None:
        preds: torch.Tensor = self(data)
        if hasattr(self, "test_metrics"):
            self.test_metrics.update(preds.softmax(dim=-1), data.y)

    def on_test_epoch_end(self) -> None:
        if hasattr(self, "test_metrics"):
            self.log_dict(self.test_metrics.compute())
            self.test_metrics.reset()

    def predict_step(self, data: Data) -> torch.Tensor:
        # if hasattr(self.model, "inference"):
        #     preds: torch.Tensor = self.model.inference(data)[:data.batch_size]
        # else:
        preds: torch.Tensor = self(data)[: data.batch_size]
        return torch.argmax(preds, dim=-1)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LambdaLR]]:
        muon_params = []
        adamw_params = []

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                # Muon: 2D+ weights (Linear, Conv)
                # AdamW: 1D params (Biases, LayerNorm, Embeddings)
                if p.ndim >= 2 and "weight" in name and "norm" not in name:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

        # Initialize Optimizers
        lr_tensor = torch.tensor(self.hparams["lr"])
        opt_muon = torch.optim.Muon(muon_params, lr=lr_tensor, adjust_lr_fn="match_rms_adamw")
        opt_adamw = torch.optim.AdamW(adamw_params, lr=lr_tensor, fused=not self.can_compile)

        # Initialize Schedulers
        sched_muon = get_cosine_schedule_with_warmup(
            optimizer=opt_muon,
            num_warmup_steps=self.hparams["warmup"],
            num_training_steps=self.hparams["max_iters"],
        )
        sched_adamw = get_cosine_schedule_with_warmup(
            optimizer=opt_adamw,
            num_warmup_steps=self.hparams["warmup"],
            num_training_steps=self.hparams["max_iters"],
        )

        return [opt_muon, opt_adamw], [sched_muon, sched_adamw]

    @staticmethod
    def _run_optimization(
        optimizers: list[LightningOptimizer], schedulers: list[LRSchedulerType] | None
    ) -> None:
        for opt in optimizers:
            opt.step()

        if schedulers is not None:
            for sch in schedulers:
                sch.step()

    def _optimization_step(
        self, optimizers: list[LightningOptimizer], schedulers: list[LRSchedulerType] | None
    ) -> None:
        if self.can_compile:
            if not hasattr(self, "_compiled_fn"):
                self._compiled_fn = torch.compile(self._run_optimization, fullgraph=False)
            self._compiled_fn(optimizers, schedulers)
        else:
            self._run_optimization(optimizers, schedulers)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)
