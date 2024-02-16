import numpy as np
import torch
from pymatgen.core import IStructure
from pymatgen.core.structure import PeriodicNeighbor


def compute_bond_angle_cosines(
    structure: IStructure,
    neighbor_lists: list[list[PeriodicNeighbor]],
    edge_features: torch.Tensor,
) -> torch.Tensor:
    """Compute the cosine of bond angles for a given structure.

    Args:
        structure (IStructure): The input structure.
        neighbor_lists (list[list[PeriodicNeighbor]]): The neighbor lists for each atom in the structure.
        edge_features (torch.Tensor): The edge features for each bond in the structure.

    Returns:
        torch.Tensor: The cosine of bond angles.
    """
    neighbors_coords = torch.from_numpy(
        np.array(
            [[x.coords for x in neighbors] for neighbors in neighbor_lists],
            dtype=np.float32,
        )
    )
    node_coords = torch.from_numpy(structure.cart_coords).float()
    node_coords = node_coords.unsqueeze(1).expand(
        len(structure), neighbors_coords.shape[1], 3
    )

    dxyz = neighbors_coords - node_coords
    r = edge_features.unsqueeze(2)

    bond_cosines = (
        dxyz @ torch.swapaxes(dxyz, 1, 2) / (r @ torch.swapaxes(r, 1, 2))
    )  # cosine rule

    return bond_cosines
