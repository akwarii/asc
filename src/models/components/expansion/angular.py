import torch
from scipy.special import sph_harm

from .radial import GaussianBasis


class RealSphHarmBasis(torch.nn.Module):
    def __init__(self):
        super().__init__()


class AngularBasisExpansion(torch.nn.Module):
    def __init__(self):
        super().__init__()