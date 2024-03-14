from pathlib import Path
from typing import Literal, TypeAlias

from torch import Tensor

AfluxResponse: TypeAlias = list[dict[str, str]] | list
PathLike: TypeAlias = str | Path

StageType: TypeAlias = Literal["fit", "validate", "test", "predict"]

SliceDictType = dict[str, Tensor | dict[str, Tensor]]
