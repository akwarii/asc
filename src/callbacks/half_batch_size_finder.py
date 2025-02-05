import lightning as L
from lightning.pytorch.callbacks import BatchSizeFinder
from lightning.pytorch.tuner.batch_size_scaling import _scale_batch_size
from lightning.pytorch.utilities.exceptions import _TunerExitException


class HalfBatchSizeFinder(BatchSizeFinder):
    """Find the largest power of 2 that can be used as the batch size then set the batch size used
    for training as half of the maximum. This is useful when graphs can be of very different sizes.

    Args:
        steps_per_trial: number of steps to run with a given batch size.
            Ideally 1 should be enough to test if an OOM error occurs,
            however in practice a few are needed.
        init_val: initial batch size to start the search with.
        max_trials: max number of increases in batch size done before
    """

    def __init__(
        self,
        steps_per_trial: int = 100,
        init_val: int = 2,
        max_trials: int = 10,
    ) -> None:
        super().__init__(steps_per_trial=steps_per_trial, init_val=init_val, max_trials=max_trials)

    def scale_batch_size(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Scales the batch size using power of 2 scaling until the largest batch size that do not
        OOM is find.
        """
        new_size = _scale_batch_size(
            trainer,
            self._mode,
            self._steps_per_trial,
            self._init_val,
            self._max_trials,
            self._batch_arg_name,
        )

        if new_size is not None:
            new_size = max(1, new_size // 2)

        self.optimal_batch_size = new_size
        if self._early_exit:
            raise _TunerExitException()
