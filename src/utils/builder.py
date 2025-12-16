from collections.abc import Callable
from typing import Any

from torch import nn
from torch_geometric.nn.resolver import normalization_resolver


def normalization_builder(
    norm: str | Callable | nn.Module | None,
    in_channels: int,
    norm_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Create a PyG normalization module for a given channel size."""
    if norm is None:
        return nn.Identity()

    if isinstance(norm, nn.Module):
        return norm

    kwargs = dict(norm_kwargs or {})
    kwargs.pop("in_channels", None)

    if isinstance(norm, str):
        layer = normalization_resolver(norm, in_channels=in_channels, **kwargs)
        return layer or nn.Identity()

    # Callable: try (in_channels, **kwargs) first, then fallback to (**kwargs)
    try:
        return norm(in_channels, **kwargs)
    except TypeError:
        return norm(**kwargs)
