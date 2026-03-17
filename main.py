#!/usr/bin/env python
import torch
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
