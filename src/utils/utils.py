from collections.abc import Mapping
from typing import Any

import pytorch_lightning as pl

from src.constants import DEFAULT_SEED


def set_seed(seed: int | str | Mapping[str, Any] = DEFAULT_SEED) -> None:
    """Set the seed for reproducibility.

    Args:
        seed: The seed value to use. If a mapping is provided, it will search for the key 'seed'.
    """
    if isinstance(seed, Mapping):
        seed = seed.get("seed", DEFAULT_SEED)

    if not isinstance(seed, int):
        if isinstance(seed, str) and seed.isdigit():
            seed = int(seed)
        else:
            seed = DEFAULT_SEED

    pl.seed_everything(seed, workers=True)
