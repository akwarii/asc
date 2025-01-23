import logging
from collections.abc import Mapping

from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class RankedLogger(logging.LoggerAdapter):
    """Logger adapter that prefixes log messages with the rank of the process.

    Args:
        name: The name of the logger. Defaults to __name__.
        rank_zero_only: Whether to log only on rank 0. Defaults to True.
        extra: Extra information to be passed to the logger. Defaults to None.
    """

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = True,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(logger=logging.getLogger(name), extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: int | None = None, *args, **kwargs) -> None:  # type: ignore[override]
        """Delegate a log call to the underlying logger, but prefix the message with the rank it's
        being processed from. If '`rank`' is provided, only log on that rank.

        Args:
            level (int): The log level.
            msg (str): The log message.
            rank (int, optional): The rank to log at.
            args: The positional arguments to be passed to the logger.
            kwargs: The keyword arguments to be passed to the logger.
        """
        if not self.isEnabledFor(level):
            return

        msg, kwargs = self.process(msg, kwargs)  # type: ignore
        current_rank = getattr(rank_zero_only, "rank", None)

        if current_rank is None:
            raise ValueError("The `rank_zero_only.rank` needs to be set before use.")

        msg = rank_prefixed_message(msg, current_rank)
        if self.rank_zero_only:
            if current_rank == 0:
                self.logger.log(level, msg, *args, **kwargs)
        else:
            if rank is None:
                self.logger.log(level, msg, *args, **kwargs)
            elif current_rank == rank:
                self.logger.log(level, msg, *args, **kwargs)
