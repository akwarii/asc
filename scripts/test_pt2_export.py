#!/usr/bin/env python3
"""Quick runtime check for a torch.export (.pt2) artifact.

This script requires the user to provide the exact .pt2 path,
loads one sample batch from the custom dataset, runs a forward pass through
the exported program, and validates basic output properties.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from src.datamodule import LightningDataset
from src.transforms.line_graph import LineGraphData

# Ensure preprocessed graph objects can be deserialized safely.
torch.serialization.add_safe_globals([LineGraphData])


def build_sample_input(data_root: str, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    datamodule = LightningDataset(
        dataset_name="custom",
        root=data_root,
        lengths=[0.7, 0.2, 0.1],
        batch_size=1,
        num_workers=0,
        use_imbalance_sampler=False,
        k=k,
    )
    datamodule.setup("fit")

    batch = next(iter(datamodule.val_dataloader()))
    x = batch.x.detach().cpu().long()
    edge_index = batch.edge_index.detach().cpu()
    edge_attr = batch.edge_attr.detach().cpu()
    return x, edge_index, edge_attr


def main() -> None:
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Validate a .pt2 export with one sample input.")
    parser.add_argument("--export", type=Path, required=True, help="Path to a .pt2 file.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="examples/silicon/data",
        help="Root directory for the custom dataset.",
    )
    parser.add_argument("--k", type=int, default=12, help="k used for processed/<k>nn dataset.")
    args = parser.parse_args()

    export_path = args.export
    print(f"Using export: {export_path}")

    # Build one sample input batch from the dataset
    x, edge_index, edge_attr = build_sample_input(args.data_root, args.k)
    print(
        "Sample shapes: "
        + (
            f"x={tuple(x.shape)}, edge_index={tuple(edge_index.shape)}, "
            f"edge_attr={tuple(edge_attr.shape)}"
        )
    )

    # Load the exported program and run inference
    exported_program = torch.export.load(str(export_path))
    exported_module = exported_program.module()

    # Run a forward pass through the exported module with the sample input
    with torch.no_grad():
        output: Any = exported_module(x, edge_index, edge_attr)

    # Basic sanity checks on the output
    if isinstance(output, (tuple, list)):
        if not output or not torch.is_tensor(output[0]):
            raise TypeError("Expected first output item to be a tensor.")
        output = output[0]
    elif not torch.is_tensor(output):
        raise TypeError(f"Expected tensor output, got {type(output).__name__}")
    if output.ndim != 2:
        raise ValueError(
            f"Expected 2D output [num_nodes, num_classes], got shape {tuple(output.shape)}"
        )
    if output.shape[0] != x.shape[0]:
        raise ValueError(
            f"Output first dimension ({output.shape[0]}) does not match num_nodes ({x.shape[0]})."
        )
    if not torch.isfinite(output).all().item():
        raise ValueError("Output contains non-finite values.")

    # Final success message with output shape
    print(f"Output shape: {tuple(output.shape)}")
    print("PT2 export verification succeeded.")


if __name__ == "__main__":
    main()
