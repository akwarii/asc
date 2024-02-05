from pathlib import Path
from typing import TypeAlias, Optional

AfluxResponse: TypeAlias = list[dict[str, str]] | list
OptionalRange: TypeAlias = Optional[tuple[int, int]]
PathLike: TypeAlias = str | Path