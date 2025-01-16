import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomPerturbation(BaseTransform):
    """Class to apply random displacement to atoms in structures.

    We mimic ASE's rattle by applying random noise to atoms (node features).

    Until further notice, we use np.random as our default RNG.

    Args:
        stddev (float): Standard deviation of the displacement (default=0.001).
        seed (int): Random seed (default=42).
        p (float): probability to apply the transform (default=0.1).
    """

    def __init__(self, stddev: float = 0.001, seed: int = 42, p: float = 0.1) -> None:
        if stddev <= 0.0:
            raise ValueError("The standard deviation must be strictly positive.")

        if p < 0.0 or p > 1.0:
            raise ValueError("The probability to displace a node must be in [0.,1.].")

        self.stddev = stddev
        self.p = p
        self.rng = torch.Generator(device="cpu").manual_seed(seed)  # TODO device

    def forward(self, x: Data) -> Data:
        """Applies random Gaussian noise to interatomic distances and angles on a batch of
        graph structures.

        Args:
            x (Sequence[Data]): batch of K-neareast neighbors graphs for periodic structures.

        Returns:
            Another batch of K-nearest neighbors graphs with Gaussian noise applied on ineratomic
            distances (`edge_dist`) and angles (`angle_cos`).
        """
        # TODO we may switch to modify the Data object every time but with a given probability
        # for every node / feature
        if torch.rand(1, generator=self.rng).item() > self.p:
            return x

        torch.normal(mean=x.edge_dist, std=self.stddev, generator=self.rng, out=x.edge_dist)
        torch.normal(mean=x.angle_cos, std=self.stddev, generator=self.rng, out=x.angle_cos)

        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stddev={self.stddev}, p={self.p})"
