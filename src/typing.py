from pathlib import Path
from typing import Any, Literal, TypeAlias

AfluxResponse: TypeAlias = list[dict[str, Any]]
PathLike: TypeAlias = str | Path

Stage: TypeAlias = Literal["fit", "validate", "test", "predict"]

FileFormats: TypeAlias = Literal[
    "cif",
    "vasp",
    "xyz",
    "lammps-dump-text",
    "lammps-data",
]
