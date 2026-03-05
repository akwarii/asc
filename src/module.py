from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import LRSchedulerPLType
from packaging.version import Version
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Data
from torchmetrics import MetricCollection

from src import models
from src.optim import get_cosine_schedule_with_warmup

MODEL_FACTORY: dict[str, Callable] = {
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

        self.can_compile = compile
        if (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability() < (7, 0)
            or Version(torch.__version__) < Version("2.0")
        ):
            print(
                "Warning: torch.compile is not supported on this device or PyTorch version. "
                "Proceeding without compilation."
            )
            self.can_compile = False

        self.save_hyperparameters(logger=False, ignore=["metrics", "compile"])
        self._create_model()

        self.criterion = F.cross_entropy

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

        self.model: torch.nn.Module = model(out_channels=out_channels, **model_kwargs)

        if self.can_compile:
            self.model = torch.compile(self.model, fullgraph=True, dynamic=True)

    def _prepare_forward_kwargs(self, data: Data) -> dict[str, Any]:
        """Extracts optional graph sampling/batching arguments from the data object."""
        return {
            "num_sampled_nodes_per_hop": getattr(data, "num_sampled_nodes_per_hop", None),
            "num_sampled_edges_per_hop": getattr(data, "num_sampled_edges_per_hop", None),
            "num_atoms": getattr(data, "num_atoms", None),
            "bond_source": getattr(data, "bond_source", None),  # Added for completeness
        }

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        **kwargs,
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            x (Tensor): The node features.
            edge_index (Tensor): The neighbor indices.
            edge_attr (Tensor): The edge features.
            kwargs (dict[str, Any]): Additional keyword arguments for the forward pass, which may
                include:
                num_sampled_nodes_per_hop (list[int] | None, optional): The number of sampled
                nodes per hop for neighbor sampling. Defaults to None.
                num_sampled_edges_per_hop (list[int] | None, optional): The number of sampled
                    edges per hop for neighbor sampling. Defaults to None.
                num_atoms (int | None, optional): The number of atoms in the graph, used for
                    batching in LineGraph. Defaults to None.
                bond_source (str | None, optional): The rows of the original graph adjacency
                    matrix. Used to reconstruct the original graph from a LineGraphData.
                    Defaults to None.
        """
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return self.model(x, edge_index, edge_attr, **clean_kwargs)

    def training_step(self, data: Data) -> Tensor:
        opts = self.optimizers()
        schs = self.lr_schedulers()

        if not isinstance(opts, list):
            opts = [opts]
        if schs is not None and not isinstance(schs, list):
            schs = [schs]

        kwargs = self._prepare_forward_kwargs(data)
        preds: Tensor = self(data.x, data.edge_index, data.edge_attr, **kwargs)
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
        kwargs = self._prepare_forward_kwargs(data)
        preds: Tensor = self(data.x, data.edge_index, data.edge_attr, **kwargs)
        loss = self.criterion(preds, torch.as_tensor(data.y))

        if hasattr(self, "val_metrics"):
            self.val_metrics.update(preds.softmax(dim=-1), data.y)
        self.log("val/loss", loss, on_epoch=True, batch_size=data.num_nodes)

    def on_validation_epoch_end(self) -> None:
        if hasattr(self, "val_metrics"):
            self.log_dict(self.val_metrics.compute())
            self.val_metrics.reset()

    def test_step(self, data: Data) -> None:
        kwargs = self._prepare_forward_kwargs(data)
        preds: Tensor = self(data.x, data.edge_index, data.edge_attr, **kwargs)
        if hasattr(self, "test_metrics"):
            self.test_metrics.update(preds.softmax(dim=-1), data.y)

    def on_test_epoch_end(self) -> None:
        if hasattr(self, "test_metrics"):
            self.log_dict(self.test_metrics.compute())
            self.test_metrics.reset()

    def predict_step(self, data: Data) -> Tensor:
        # if hasattr(self.model, "inference"):
        #     preds: Tensor = self.model.inference(data)[:data.batch_size]
        # else:
        kwargs = self._prepare_forward_kwargs(data)
        preds: Tensor = self(data.x, data.edge_index, data.edge_attr, **kwargs)
        return torch.argmax(preds, dim=-1)

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LambdaLR]]:
        muon_params = []
        adamw_params = []

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                # Muon: 2D+ weights (Linear, Conv)
                # AdamW: 1D params (Biases, Norms, Embeddings)
                if p.ndim >= 2 and "weight" in name and "norm" not in name:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

        # Initialize Optimizers
        # TODO torch.compile does not support Muon optimizer yet
        # we will set lr as a tensor once it is supported to avoid graph breaks
        lr = self.hparams["lr"]
        opt_muon = torch.optim.Muon(muon_params, lr=lr, adjust_lr_fn="match_rms_adamw")
        opt_adamw = torch.optim.AdamW(adamw_params, lr=lr)

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
        optimizers: list[LightningOptimizer], schedulers: list[LRSchedulerPLType] | None
    ) -> None:
        for opt in optimizers:
            opt.step()

        if schedulers is not None:
            for sch in schedulers:
                sch.step()

    def _optimization_step(
        self, optimizers: list[LightningOptimizer], schedulers: list[LRSchedulerPLType] | None
    ) -> None:
        # TODO torch.compile does not support Muon optimizer yet
        # if self.can_compile:
        #     if not hasattr(self, "_compiled_fn"):
        #         self._compiled_fn = torch.compile(self._run_optimization, fullgraph=False)
        #     self._compiled_fn(optimizers, schedulers)
        # else:
        self._run_optimization(optimizers, schedulers)
