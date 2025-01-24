from pathlib import Path
from typing import Literal, TypeAlias

AfluxResponse: TypeAlias = list[dict[str, str]]
PathLike: TypeAlias = str | Path

Stage: TypeAlias = Literal["fit", "validate", "test", "predict"]
