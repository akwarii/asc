from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logging.getLogger(__name__)


# FIXME: The logs are not used correctly, only the root logger in the main file is used
def set_log_handles(level: int, log_path: str | Path | None = None):
    logger = logging.getLogger("cegann")
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handler and set level to debug
    if log_path:
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
