#!/usr/bin/env python
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from src.datamodule import LightningDataset
from src.module import Module
from src.transforms.line_graph import LineGraphData

# Ensure that LineGraphData is treated as a safe global for torch.load
torch.serialization.add_safe_globals([LineGraphData])


class CustomLightningCLI(LightningCLI):
    """Custom LightningCLI to handle dynamic max_iters calculation and argument linking."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Link num_classes from DataModule to Module automatically."""
        parser.link_arguments(
            "data.num_classes", "model.model.init_args.out_channels", apply_on="instantiate"
        )

    def before_fit(self) -> None:
        """Calculate max_iters for the optimizer/scheduler before training starts."""
        # If max_iters is set to -1 or 0, we calculate it dynamically
        if self.model.hparams.get("max_iters", 0) <= 0:
            # We need to setup the datamodule to get the number of batches
            self.datamodule.setup("fit")
            train_loader = self.datamodule.train_dataloader()

            # Use trainer.max_epochs and check for gradient accumulation
            max_epochs = self.trainer.max_epochs
            if max_epochs is None or max_epochs == -1:
                # Default to a large number if not specified, though usually it is
                max_epochs = 1000

            num_batches = len(train_loader)
            accumulate_grad_batches = getattr(self.trainer, "accumulate_grad_batches", 1)

            max_iters = (num_batches // accumulate_grad_batches) * max_epochs

            # Update the model hyper-parameters
            self.model.hparams.max_iters = max_iters
            print(
                f"INFO: Calculated max_iters: {max_iters} "
                f"({num_batches} batches * {max_epochs} epochs)"
            )

    def _make_export_example(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one representative (x, edge_index, edge_attr) tuple for export."""
        self.datamodule.setup("fit")
        batch = next(iter(self.datamodule.val_dataloader()))
        return (
            batch.x.detach().cpu().long(),
            batch.edge_index.detach().cpu(),
            batch.edge_attr.detach().cpu(),
        )

    def _load_model_for_export(self, ckpt_path: Path) -> torch.nn.Module:
        """Load a checkpoint for export and always return an uncompiled raw model."""
        try:
            module = Module.load_from_checkpoint(
                str(ckpt_path),
                map_location="cpu",
                compile=False,
            )
            raw_model = module.model
        except RuntimeError as exc:
            # Checkpoints produced with compile=True may store wrapped key names.
            if "_orig_mod" not in str(exc):
                raise
            module = Module.load_from_checkpoint(
                str(ckpt_path),
                map_location="cpu",
                compile=True,
            )
            raw_model = module.model
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod

        if hasattr(raw_model, "_orig_mod"):
            raw_model = raw_model._orig_mod

        return raw_model.eval().cpu()

    def after_fit(self) -> None:
        """Export the best checkpoint with torch.export using an uncompiled model."""
        # Find the best checkpoint path from the trainer callbacks
        ckpt_path = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint) and callback.best_model_path:
                ckpt_path = Path(callback.best_model_path)
        if ckpt_path is None or not ckpt_path.exists():
            raise RuntimeError("No valid checkpoint found for export.")

        raw_model = self._load_model_for_export(ckpt_path)

        # Handling dynamic shapes for graph data
        num_nodes = torch.export.Dim("num_nodes", min=2)  # I think we never have just 1 node?
        num_edges = torch.export.Dim("num_edges", min=2)  # Graphs are directed, so >= 2 edges?
        dynamic_shapes = {
            "x": {0: num_nodes},
            "edge_index": {1: num_edges},
            "edge_attr": {0: num_edges},
        }

        # Construct one representative input example for export
        example = self._make_export_example()

        print("Exporting best checkpoint with torch.export...")
        exported = torch.export.export(
            raw_model,
            example,
            dynamic_shapes=dynamic_shapes,
        )

        out_dir = Path("exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        export_path = out_dir / f"{ckpt_path.stem}.pt2"
        torch.export.save(exported, str(export_path))
        print(f"torch.export artifact written to: {export_path}")


def main() -> None:
    """Main entrypoint."""
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    CustomLightningCLI(
        model_class=Module,
        datamodule_class=LightningDataset,
        seed_everything_default=42,
        parser_kwargs={"parser_mode": "yaml"},
        auto_configure_optimizers=False,
    )


if __name__ == "__main__":
    main()
