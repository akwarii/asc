from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers.param_scheduler import LRScheduler
from ignite.metrics import Accuracy
from torch.optim.lr_scheduler import StepLR

from dataloader import get_train_val_test_loader
from logger import set_log_handles
from model import CEGANN
from utils import (load_dataset, load_settings, prepare_batch_fn,
                   resume_training, save_checkpoint)

# logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validate_args():
    try:
        training_data = sys.argv[1]
    except IndexError:
        logging.error("Training data folder not provided")
        raise IndexError(
            "The training data folder must be provided as first argument"
        )

    try:
        checkpoint_dir = sys.argv[2]
    except IndexError:
        checkpoint_dir = "model_checkpoints"
        logging.warning(
            f"Checkpoint directory not provided, using default: {checkpoint_dir}")

    try:
        logfile = sys.argv[3]
    except IndexError:
        logfile = "log.model"
        logging.warning(f"Log file not provided, using default: {logfile}")

    if not Path(training_data).is_dir():
        raise FileNotFoundError("Training data not found")

    return training_data, checkpoint_dir, logfile


def run(
    model,
    train_loader,
    val_loader,
    test_loader,
    settings,
    output_dir: Path | str = Path.cwd() / "model",
    screen_log="log.model",
):
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    global best_val_accuracy
    global total_batch_loss
    global iteration_count

    best_val_accuracy, total_batch_loss, iteration_count = -1e300, 0, 0

    # -----------losss--------------------
    loss = torch.nn.CrossEntropyLoss()
    val_metrics = {
        "accuracy": Accuracy(),
    }

    # ---------model----------------
    model.to(device)

    if settings.optimizer == "adam":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        )

    if settings.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=settings.learning_rate,
            momentum=settings.momentum,
        )

    trainer = create_supervised_trainer(
        model,
        optimizer,
        loss,
        prepare_batch=prepare_batch_fn,
        device=device,
    )

    if settings.scheduler:
        torch_lr_scheduler = StepLR(
            optimizer, step_size=settings.step_size, gamma=settings.gamma
        )
        scheduler = LRScheduler(torch_lr_scheduler)
        trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

    if settings.progress:
        print("Progress bar enabled")
        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    evaluator = create_supervised_evaluator(
        model,
        prepare_batch=prepare_batch_fn,
        metrics=val_metrics,
        device=device,
    )

    test_evaluator = create_supervised_evaluator(
        model,
        prepare_batch=prepare_batch_fn,
        metrics=val_metrics,
        device=device,
    )

    if settings.resume:
        resume_training(output_dir, model, optimizer,
                        scheduler, trainer, settings)
    else:
        Path(screen_log).unlink(missing_ok=True)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global total_batch_loss
        global iteration_count
        total_batch_loss += engine.state.output
        iteration_count += 1

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        global best_val_accuracy
        global total_batch_loss
        global iteration_count

        epoch = engine.state.epoch

        avg_batch_loss = total_batch_loss / iteration_count
        total_batch_loss, iteration_count = 0, 0

        epoch = engine.state.epoch

        evaluator.run(val_loader)

        valmetrics = evaluator.state.metrics
        val_accuracy = valmetrics["accuracy"]

        if len(test_loader) != 0:
            test_evaluator.run(test_loader)
            testmetrics = test_evaluator.state.metrics
            test_accuracy = testmetrics["accuracy"]

        if epoch % settings.checkpoint_every == 0:
            save_checkpoint(model, optimizer, scheduler, trainer,
                            best_val_accuracy, epoch, is_best=False, path=output_dir)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            logging.info(
                f"Saving best checkpoint with {best_val_accuracy} accuracy (epoch {epoch})")
            save_checkpoint(model, optimizer, scheduler, trainer,
                            best_val_accuracy, epoch, is_best=True, path=output_dir)

        if settings.progress:
            pbar_log_msg = f"train_loss:{avg_batch_loss:.4f},  val_acc: {val_accuracy:.4f}"
            if test_loader:
                pbar_log_msg += f",  test_acc: {test_accuracy:.4f}"

            pbar.log_message(pbar_log_msg)

        screen_log_msg = f"{avg_batch_loss:.4f},{val_accuracy:.4f}"
        if len(test_loader) != 0:
            screen_log_msg += f",{test_accuracy:.4f}"

        with open(screen_log, "a") as outfile:
            outfile.write(screen_log_msg+"\n")

        epoch += 1

    trainer.run(train_loader, max_epochs=settings.epochs)
    pbar.close()


def main() -> None:
    training_data, checkpoint_dir, logfile = validate_args()

    # set_log_handles(logging.DEBUG, logfile)
    logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode="w")

    settings = load_settings()
    graphs = load_dataset(training_data, settings)
    
    if graphs.num_classes != settings.n_classes:
        logging.warning(
            f"Number of classes in dataset ({graphs.num_classes}) does not match n_classes ({settings.n_classes}). Setting n_classification to {graphs.num_classes}."
        )
        settings.n_classes = graphs.num_classes

    train_loader, val_loader, test_loader = get_train_val_test_loader(
        graphs,
        collate_fn=graphs.collate,
        batch_size=settings.batch_size,
        train_ratio=settings.train_ratio,
        val_ratio=settings.val_ratio,
        test_ratio=settings.test_ratio,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        train_size=settings.train_size,
        test_size=settings.test_size,
        val_size=settings.val_size,
    )

    model = CEGANN(
        settings.gbf_bond,
        settings.gbf_angle,
        n_conv_edge=settings.n_conv_edge,
        edge_expansion_units=settings.h_fea_edge,
        angle_expansion_units=settings.h_fea_angle,
        n_classes=settings.n_classes,
        pooling=settings.pooling,
        embedding=settings.embedding,
    )

    try:
        model = torch.compile(model)
    except Exception:
        logging.warning(
            "Model is not compilable. Consider upgrading to PyTorch 2 or higher.")
    else:
        print("Model was compiled successfully.")
        logging.info("Model was compiled successfully.")

    run(
        model,
        train_loader,
        val_loader,
        test_loader,
        settings,
        output_dir=checkpoint_dir,
        screen_log=logfile
    )


if __name__ == "__main__":
    main()
