import numpy as np
from pymatgen.core import IStructure
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import PeriodicNeighbor


def find_knn_in_shell(
    structure: IStructure,
    rcut: float,
    n_neighbors: int,
    delta: float = 1,
) -> list[list[PeriodicNeighbor]]:
    if rcut <= 0:
        species = [site.specie.symbol for site in structure.sites]
        rcut = max([Element(elm).atomic_radius * 3 for elm in species])

    all_neighbors = structure.get_all_neighbors(rcut, include_index=True)
    len_neighbors = np.array([len(neighbor) for neighbor in all_neighbors])
    missing_neighbors_idx = np.where((len_neighbors < n_neighbors))[0]

    for i in missing_neighbors_idx:
        rcut = rcut
        n_neighbors = len(all_neighbors[i])
        while n_neighbors < n_neighbors:
            rcut += delta
            neighbor = structure.get_neighbors(structure[i], rcut)
            n_neighbors = len(neighbor)
        all_neighbors[i] = neighbor

    all_neighbors_sorted = [
        sorted(neighbors, key=lambda x: x[1])[:n_neighbors]
        for neighbors in all_neighbors
    ]
    
    return all_neighbors_sorted
