import torch.nn as nn


class BaseModel(nn.Module):
    """Base model class that all models should inherit from."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
