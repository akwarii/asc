from pathlib import Path
from typing import Literal, TypeAlias

AfluxResponse: TypeAlias = list[dict[str, str]] | list
PathLike: TypeAlias = str | Path

StageType: TypeAlias = Literal["fit", "validate", "test", "predict"]
