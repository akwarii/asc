from pathlib import Path
from typing import TypeAlias

AfluxResponse: TypeAlias = list[dict[str, str]] | list
PathLike: TypeAlias = str | Path
