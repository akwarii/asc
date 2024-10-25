from collections.abc import Generator, Sequence
from pymatgen.core import Structure
from src.processing.graph import KNNGraph
import numpy as np
import torch
import itertools
from torch_geometric.data import Data

class RandomDisplacement(torch.nn.Module):
    """Class to apply random displacement to atoms in structures.
    
    We mimic ASE's rattle by applying random noise to atoms (node features).

    Until further notice, we use np.random as our default RNG.
    
    Args:
        stddev (float): Standard deviation of the displacement (default=0.001).
        seed (int): Random seed (default=42).
    """
    def __init__(
            self, 
            stddev: float = 0.001, 
            seed: int = 42,
    ) -> None:
        super().__init__()
        self.stddev = stddev
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)

    def forward(
        self, 
        x: Sequence[Data],
        progress_bar: bool=True
    ) -> Sequence[Data] :
        """Applies random Gaussian noise to atomic positions on a batch of graph structures.
        This means modifying atomic postions (graph node features) as well as distances (graph
        edges features).

        Args:
            x (Sequence[Data]): batch of K-neareast neighbors graphs for periodic structures.
            progress_bar (bool): whether we show progression with tqdm pbar() (default=True).
        Returns:
            Another batch of K-nearest neighbors graphs with Gaussian noise applied on atomic
            positions (node features) and distances (edge features).
        """

        if progress_bar :
            from tqdm import tqdm
            pbar = tqdm(total=len(list(x)), # fastest according to A. Bogdanov and mkrieger-1 on https://stackoverflow.com/questions/393053/length-of-generator-output
                        desc="Augmenting data (Gaussian noise)")
            
        rattled_graphs = []

        for graph in x:

            # Actual random displacement to apply to atoms (nodes features)
            displacements = self.rng.normal(scale=self.stddev, size=graph.pos.size())

            # Updated nodes positions
            pos = graph.pos + displacements

            # Useful PBC translation vectors
            trans = self._pbc_preparation(graph.cell)

            # Computing distances
            distances = torch.zeros(size=graph.edge_dist.size())
            for i, at_idx in enumerate(graph.edge_index[0]) :
                neigh_idx = graph.edge_index[1,i]
                # Distances in all closest periodic images
                d = self._distance(pos[at_idx], pos[neigh_idx], trans)
                # We identify which computed PBC distance corresponds (ie.
                # is the closest) to the distance before rattling positions.
                distances[i] = d[torch.argmin(
                    torch.abs(d-graph.edge_dist[i])
                )].item()
                #################################################################
                # NB: as much as I would like to, I cannot efficiently refactor #
                #     above to better employ all other distances stored in `d`. #
                #     Indeed, it is possible to identify all indexes related to #
                #     [at_idx, neigh_idx] atoms, and we can update them at once #
                #     but as it KNN graphs, there are many cases where J belong #
                #     to I's neighbors but the opposite is not True.            #
                #     Therefore, we have to check every distances individually, #
                #     and among all the trials I have made the "naive" approach #
                #     from above is (numerically) the least expensive.          #
                #     If one thinks of another clever way to improve this, with #
                #     numerical performance improvements, I would be curious to #
                #     see it in detail. DB                                      #
                ###############################################################
            rattled_graphs.append(
                Data(
                    num_nodes=graph.num_nodes,
                    pos=pos,
                    cell=graph.cell,
                    edge_index=graph.edge_index,
                    edge_dist=distances   
                )
            )

            # yield Data(
            #     num_nodes=graph.num_nodes,
            #     pos=pos,
            #     cell=graph.cell,
            #     edge_index=graph.edge_index,
            #     edge_dist=distances   
            # )

            if progress_bar :
                pbar.update(1)

        if progress_bar :
             pbar.close()

        return rattled_graphs
    
    @staticmethod
    def _pbc_preparation(
        cell : torch.Tensor
    ) -> torch.Tensor :
        """Returns all the translations to apply to positions in order to
        get images from periodic images of cells adjacent to the main one.

        Args:
            cell (torch.Tensor): 2D tensor with all 3 lattice vectors
        Returns:
            trans (torch.Tensor): 2D tensor with all translations to apply to positions
                                  in order to get periodic images.
        """
        latvec_multiplier = torch.FloatTensor(
            [m for m in itertools.product(torch.arange(3)-1, repeat=3)]
        )
        return torch.sum( 
            torch.transpose(
                torch.transpose(
                    cell.unsqueeze(0).repeat(27,1,1),
                    dim0=1, dim1=2
                ) * latvec_multiplier[:,None],
                dim0=1, dim1=2
            ), dim=1
        )
    
    @staticmethod
    def _distance(
        atom_pos: torch.Tensor,
        neigh_pos: torch.Tensor,
        trans: torch.Tensor
    ) -> float :
        """Given two position vectors and translations to apply to get
        periodic images, returns all distances (up-to 1st PBC images) 
        between the central atom and all images of the neighbor.

        Args:
            atom_pos (torch.Tensor): 1D vector for the central atom positions
            neigh_pos (torch.Tensor): 1D vector for the neighbor positions
            trans (torch.Tensor): 3D tensor for the translations to apply for periodic images
        Returns:
            (float) The distance to the closest image of the neighbor
        """
        ne_trans = neigh_pos.unsqueeze(0).repeat(27,1) + trans
        return torch.nn.functional.pairwise_distance(
            atom_pos * torch.ones([27,3]), ne_trans
        )
    
    def rattle_box(
        self,
        box_str: str
    ) -> str :
        """Applies ASE random displacement `Atoms.rattle()` to a system.
        This function is not used for now but could be useful to pre-aumgent
        data and write it as POSCAR files.
        
        Args:
            box_str : a string in the POSCAR format.
        Returns:
            Another string in the POSCAR format with rattled atomic positions.
        """
        from ase import Atoms
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        try :
            at = Structure.from_str(box_str, fmt="poscar").to_ase_atoms()
            at.rattle(
                stdev=self.stddev,
                seed=self.seed
            )
        except IndexError :
            print("The file {box_str} was not a POSCAR and has not been augmented.")
            return None
        return AseAtomsAdaptor.get_structure(at).to(fmt="poscar")

class RandomExpansion(torch.nn.Module):
    """Class to apply random expansion to boxes.
    
    Randomly increases/shrinks by multiplying cell (and hence volumes)
    and positions (and hence distances) by a random number following a
    normal distribution with 95% of values in [-2*std, 2*std].

    Until further notice, we use np.random as our default RNG.
    
    Args:
        stddev (float): Standard deviation of the displacement (default=0.05
                        to have 95% of expansions/shrinkings smaller than 1%).
        seed (int): Random seed (default=42).
    """
    def __init__(
            self,
            stddev: float = 0.05, 
            seed: int = 42,
    ) -> None:
        super().__init__()
        self.stddev = stddev
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)

    def forward(
        self, 
        x: Sequence[Data],
        progress_bar: bool=True
    ) -> Sequence[Data] :
        
        if progress_bar :
            from tqdm import tqdm
            pbar = tqdm(total=len(list(x)), # fastest according to A. Bogdanov and mkrieger-1 on https://stackoverflow.com/questions/393053/length-of-generator-output
                        desc="Augmenting data (random expansion)")
        
        random_scale = self.rng.normal(scale=self.stddev, size=len(x), loc=1.)
        scaled_graphs = []

        for i, graph in enumerate(x):
            scaled_graphs.append(
                Data(
                    num_nodes=graph.num_nodes,
                    pos=graph.pos * random_scale[i],
                    cell=graph.cell * random_scale[i],
                    edge_index=graph.edge_index,
                    edge_dist=graph.edge_dist * random_scale[i]
                )
            )

            if progress_bar :
                pbar.update(1)

        if progress_bar :
             pbar.close()

        return scaled_graphs

class RandomNodeDrop(torch.nn.Module):
    """Class to apply random node dropout to boxes.
    
    Randomly drops every node of a graph with probability p.
    This mimics crystalline defects in units cells (without
    relaxation).
    
    Args:
        p (float): dropout probability in [0.,1.] (default=0.05).
        seed (int): Random seed (default=42).
    """
    def __init__(
            self, 
             p: float = 0.05, 
            seed: int = 42,
    ) -> None:
        super().__init__()
        self.p = p
        self.seed = seed
        self.rng = np.random.RandomState(seed=self.seed)

    def forward(
        self, 
        x: Sequence[Data],
        keep_undropped: bool=False,
        progress_bar: bool=True
    ) -> Sequence[Data] :
        
        if progress_bar :
            from tqdm import tqdm
            pbar = tqdm(total=len(list(x)), # fastest according to A. Bogdanov and mkrieger-1 on https://stackoverflow.com/questions/393053/length-of-generator-output
                        desc="Augmenting data (random node drop)")
        
        dropped_graphs = []

        for graph in x:

            randDrops = self.rng.choice([0,1],
                                        size=graph.num_nodes,
                                        p=[self.p, 1.-self.p]) # 0 for drop, 1 for keep
            dropout = np.where(randDrops==0)[0]
            
            if not any(dropout) :
                if keep_undropped :
                    dropped_graphs.append(
                        Data(
                            num_nodes=graph.num_nodes,
                            pos=graph.pos,
                            cell=graph.cell,
                            edge_index=graph.edge_index,
                            edge_dist=graph.edge_dist
                        )
                    ) 
                if progress_bar : pbar.update(1)
                continue # No dropout to apply, no need to duplicate the graph

            # Updated nodes/positions
            nodes_tokeep = np.where(randDrops)[0]
            pos = graph.pos[nodes_tokeep]

            # Need to remove edges involving the dropped nodes
            edges_tokeep = []
            for edge in range(graph.edge_index.size()[0]) :
                if all(idx in nodes_tokeep for idx in graph.edge_index[edge]) :
                    edges_tokeep.append(edge)
            edge_index = graph.edge_index[edges_tokeep, :]
            edge_dist = graph.edge_dist[edges_tokeep]

            dropped_graphs.append(
                Data(
                    num_nodes=graph.num_nodes - np.size(dropout),
                    pos=pos,
                    cell=graph.cell,
                    edge_index=edge_index,
                    edge_dist=edge_dist
                )
            )

            if progress_bar :
                pbar.update(1)

        if progress_bar :
             pbar.close()

        return dropped_graphs

# TODO (maybe not?) implement molecular dynamics data augmentation
# DB : https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md
# DB : after thoughts/discussions, problems to identify interatomic
#      potentials and/or numerical cost of AIMD calculations make
#      MolecularDynamics too complicated/expensive to be massively
#      applied to large and varied batches of data.
class MolecularDynamics(torch.nn.Module):
    pass