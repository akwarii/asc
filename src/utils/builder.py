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


def class_instantiator(obj: Any, **kwargs) -> Any:  # noqa: ANN401
    """Instantiate a class from a variety of formats (class type, dict, Namespace, list).

    Args:
        obj: The object to instantiate. Can be:
            - An already instantiated object. Returned as is.
            - A list of objects. Recursively instantiates each item.
            - A class type. Instantiated with `**kwargs`.
            - A dict with 'class_path'. Dynamically imported and instantiated with merged
                `init_args` and `kwargs`.
            - A Namespace with 'class_path'. Behaves like a dict.
        **kwargs: Default arguments to pass to the constructor. These are merged with any
            `init_args` found in the object configuration, with `init_args` taking precedence
            (unless they are None/missing).
    """
    if obj is None:
        return None

    if isinstance(obj, list):
        return [class_instantiator(item, **kwargs) for item in obj]

    # Handle class types (passed from python directly)
    if isinstance(obj, type):
        return obj(**kwargs)

    # Handle Namespace or dict (passed from CLI)
    class_path = None
    init_args: dict[str, Any] = {}

    # Namespace
    if hasattr(obj, "class_path"):
        class_path = obj.class_path
        init_args = vars(obj.init_args) if obj.init_args else dict()

    # Dict
    elif isinstance(obj, dict) and "class_path" in obj:
        class_path = obj["class_path"]
        init_args = obj.get("init_args", {})

    # If we have a class_path, we try to import and instantiate it
    if class_path:
        import importlib

        module_name, class_name = class_path.rsplit(".", 1)

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        # If init_args has a value, we use it. If not, we fall back to kwargs.
        merged_kwargs = init_args.copy()
        merged_kwargs.update(kwargs)

        return cls(**merged_kwargs)

    # Fallback:  we assume it's already instantiated
    return obj
