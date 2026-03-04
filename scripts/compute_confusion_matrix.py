"""
Script to load a checkpoint and compute confusion matrix for a dataset.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Trainer, seed_everything
from sklearn.metrics import classification_report, confusion_matrix
from src import LightningDataset, Module
from src.constants import DEFAULT_SEED
from tqdm.auto import tqdm


@torch.inference_mode()
def evaluate_model(
    model: Module,
    dataloader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataloader and return predictions and ground truth labels.

    Args:
        model: The trained model
        dataloader: DataLoader to evaluate on
        device: Device to run evaluation on

    Returns:
        Tuple of (predictions, ground_truth) as numpy arrays
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        batch = batch.to(device)

        # Forward pass
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16):
            logits = model(batch)

        # Get predictions
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch.y.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_preds)
    ground_truth = np.concatenate(all_labels)

    return predictions, ground_truth


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    save_path: Path | None = None,
    normalize: bool = True,
) -> None:
    """
    Plot confusion matrix using seaborn.

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".4f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(12, 10))

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    # Plot confusion matrix
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    # Set ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )

    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = format(cm[i, j], fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a checkpoint and compute confusion matrix")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="custom",
        help="Dataset name (default: custom)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Which split to evaluate on (default: test). "
        + "Use 'all' for the entire dataset without splitting",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=20,
        help="Number of neighbors for graph construction (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for evaluation (default: 4096)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of workers for data loading (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./confusion_matrices",
        help="Directory to save confusion matrix plots (default: ./confusion_matrices)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize the confusion matrix",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions and ground truth to file",
    )

    args = parser.parse_args()

    # Setup
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    seed_everything(DEFAULT_SEED)

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache.")

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load model
    trainer = Trainer(
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    with trainer.init_module(empty_init=True):
        model = Module.load_from_checkpoint(str(checkpoint_path))

    print("Model loaded successfully.")

    # Load dataset
    print(f"Loading dataset: {args.dataset}")

    # Set lengths based on split choice
    if args.split == "all":
        lengths = (1.0, 0.0, 0.0)  # All data in train split
        print("Using entire dataset without splitting")
    else:
        lengths = (0.7, 0.2, 0.1)

    datamodule = LightningDataset(
        dataset_name=args.dataset,
        lengths=lengths,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        k=args.num_neighbors,
        use_imbalance_sampler=False,  # No sampling for evaluation
        force_reload=False,
    )

    print("Dataset loaded successfully.")
    print("Number of classes:", datamodule.num_classes)

    # Setup datamodule to access dataloaders
    datamodule.setup("test")

    # Get the appropriate dataloader
    if args.split == "all" or args.split == "train":
        dataloader = datamodule.train_dataloader()
    elif args.split == "val":
        dataloader = datamodule.val_dataloader()
    else:
        dataloader = datamodule.test_dataloader()

    print(f"Evaluating on {args.split} split...")

    # Evaluate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictions, ground_truth = evaluate_model(model, dataloader, device)

    print("\nEvaluation complete!")
    print(f"Total samples: {len(predictions)}")
    print(f"Unique classes in predictions: {np.unique(predictions)}")
    print(f"Unique classes in ground truth: {np.unique(ground_truth)}")

    # Compute confusion matrix
    cm = confusion_matrix(ground_truth, predictions)

    # Get class names if available
    class_names = None
    if hasattr(datamodule, "class_names"):
        class_names = datamodule.class_names
    elif hasattr(datamodule.dataset, "class_names"):
        class_names = datamodule.dataset.class_names

    # Print classification report
    print("\n" + "=" * 80)
    print("Classification Report:")
    print("=" * 80)
    report = classification_report(
        ground_truth,
        predictions,
        target_names=class_names,
        digits=4,
    )
    print(report)

    # Save classification report
    report_path = output_dir / f"classification_report_{args.split}.txt"
    with open(report_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Total samples: {len(predictions)}\n\n")
        f.write(report)
    print(f"Classification report saved to {report_path}")

    # Plot and save confusion matrices
    # 1. Raw counts
    cm_raw_path = output_dir / f"confusion_matrix_{args.split}_raw.png"
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        save_path=cm_raw_path,
        normalize=False,
    )

    # 2. Normalized
    if not args.no_normalize:
        cm_norm_path = output_dir / f"confusion_matrix_{args.split}_normalized.png"
        plot_confusion_matrix(
            cm,
            class_names=class_names,
            save_path=cm_norm_path,
            normalize=True,
        )

    # Save raw confusion matrix
    cm_npy_path = output_dir / f"confusion_matrix_{args.split}.npy"
    np.save(cm_npy_path, cm)
    print(f"Confusion matrix array saved to {cm_npy_path}")

    # Save predictions if requested
    if args.save_predictions:
        pred_path = output_dir / f"predictions_{args.split}.npz"
        np.savez(
            pred_path,
            predictions=predictions,
            ground_truth=ground_truth,
        )
        print(f"Predictions saved to {pred_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
