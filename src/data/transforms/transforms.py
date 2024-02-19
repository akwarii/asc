import torch

__all__ = [
    "Compose",
    "Normalize",
]

class Compose:
    """Reimplementation of the `torchvision.transforms.Compose` class.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     transforms.Normalize(0.1307, 0.3081),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_object):
        for t in self.transforms:
            data_object = t(data_object)
        return data_object

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
    
    
class Normalize(torch.nn.Module):
    """Normalize a tensor with mean and standard deviation.
    This implementation is based on the `torchvision.transforms.Normalize` class
    but is adapted to non-image objects.
    
    Args:
        mean (float): mean of the tensor.
        std (float): standard deviation of the tensor.
        inplace (bool): apply the operation in-place. Default to False.
        device (str): device to use for the operation.
        
    Methods:
        forward: apply the normalization to the tensor.
        state_dict: return the state of the normalization transformation.
        load_state_dict: load the state of the normalization transformation.
    """
    def __init__(
        self,
        mean: float | None = None,
        std: float | None = None,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        
        if mean is None or std is None:
            raise ValueError("Mean and std should be provided.")
        
        self.mean = mean
        self.std = std
            
        self.inplace = inplace
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the normalization to the tensor.
        
        Args:
            tensor (torch.Tensor): input tensor to normalize.
            
        Returns:
            torch.Tensor: normalized tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()
        
        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        
        if (std == 0).any():
            raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
        
        return tensor.sub_(mean).div_(std)
    
    def state_dict(self) -> dict[str, float]:
        """Return the state of the normalization transformation."""
        return {"mean": self.mean, "std": self.std}
    
    def load_state_dict(self, state_dict: dict[str, float]) -> None:
        """Load the state of the normalization transformation.
        
        Args:
            state_dict (dict): normalization state. Should contain the keys "mean" and "std".
        """
        self.mean = state_dict.get("mean")
        self.std = state_dict.get("std")
        
        if self.mean is None or self.std is None:
            raise ValueError("State dict should contain the keys 'mean' and 'std'.")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"