import itertools
from collections.abc import Sequence

import numpy as np
import torch
from torch_geometric.data import Data


# TODO review this class
# TODO forward must return a Data object with tensors of dtype float32
class RandomDisplacement(torch.nn.Module):
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
        super().__init__()
        self.stddev = stddev
        self.p = p
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def forward(self, x: Data) -> Data:
        """Applies random Gaussian noise to interatomic distances and angles on a batch of
        graph structures.

        Args:
            x (Sequence[Data]): batch of K-neareast neighbors graphs for periodic structures.

        Returns:
            Another batch of K-nearest neighbors graphs with Gaussian noise applied on ineratomic
            distances (`edge_dist`) and angles (`angle_cos`).
        """
        ############################ NOTE ################################
        # As positions are not currently used in the training, and because
        # the idea is to slightly perturb distances and angles only, we
        # don't really need to worry about each distances and angles being
        # consistent between the same images of a same atom. Also, for KNN
        # which are not two-way (j can be in i's neighborhood but i is not
        # necessarily in j's) , we don't have to worry about rij = rji and
        # theta_ijk = theta_jki: as long as they are close enough it won't
        # be a huge issue. An exact implementation can be found in:
        #     `self.forward_exact()`
        # However, please note the exact implementation is absurdly longer
        # (4h+ on the `Materials-Project`` dataset, instead of 40+ seconds
        # on my laptop). ~ DB
        ######################### END OF NOTE ############################

        # TODO we may switch to modify the Data object every time but with a given probability
        # for every node / feature
        if torch.rand(1, generator=self.rng).item() > self.p:
            return x

        distances = torch.normal(mean=x.edge_dist, std=self.stddev, generator=self.rng)
        angle_cos = torch.normal(mean=x.angle_cos, std=self.stddev, generator=self.rng)

        augmented_data = Data(
            num_nodes=x.num_nodes,
            pos=x.pos,
            cell=x.cell,
            edge_index=x.edge_index,
            edge_dist=distances,
            angle_cos=angle_cos,
        )

        return augmented_data

    def forward_exact(self, x: Sequence[Data], progress_bar: bool = True) -> Sequence[Data]:
        """Applies random Gaussian noise to atomic positions on a batch of graph structures.
        This means modifying atomic positions (graph node features) as well as distances (graph
        edges features).

        Args:
            x (Sequence[Data]): batch of K-neareast neighbors graphs for periodic structures.
            progress_bar (bool): whether we show progression with tqdm pbar() (default=True).

        Returns:
            Another batch of K-nearest neighbors graphs with Gaussian noise applied on atomic
            positions (node features) and distances (edge features).
        """
        if progress_bar:
            from tqdm import tqdm

            pbar = tqdm(
                total=len(list(x)),
                desc="Augmenting data (Gaussian noise)",
            )

        rattled_graphs = []

        for graph in x:
            # Actual random displacement to apply to atoms (nodes features)
            displacements = self.rng.normal(scale=self.stddev, size=graph.pos.size())

            # Updated nodes positions
            pos = graph.pos + displacements

            # Useful PBC translation vectors
            trans = self._pbc_preparation(graph.cell)

            # Computing distances
            distances = torch.zeros(size=graph.edge_dist.size())
            for i, at_idx in enumerate(graph.edge_index[0]):
                # Distances in all closest periodic images
                d = self._distance(pos[at_idx], pos[graph.edge_index[1, i]], trans)
                # We identify which computed PBC distance corresponds (ie.
                # is the closest) to the distance before rattling positions.
                distances[i] = d[torch.argmin(torch.abs(d - graph.edge_dist[i]))].item()
                #################################################################
                # NB: as much as I would like to, I cannot efficiently refactor #
                #     above to better employ all other distances stored in `d`. #
                #     Indeed, it is possible to identify all indexes related to #
                #     [at_idx, neigh_idx] atoms, and we can update them at once #
                #     but with KNN graphs, there are many cases where J belongs #
                #     to I's neighbors but the opposite is not True.            #
                #     Therefore, we have to check every distances individually, #
                #     and among all the trials I have made the "naive" approach #
                #     from above is (numerically) the least expensive.          #
                #     If one thinks of another clever way to improve this, with #
                #     numerical performance improvements, I would be curious to #
                #     see it in detail. DB                                      #
                #################################################################

            # Angles cosine
            m = pos.size()[0]  # Number of atoms
            k = graph.edge_index[1].size()[0] // m  # Number of nearest neighbors
            _nbr_idx = torch.reshape(torch.LongTensor(graph.edge_index[1]), (m, k))
            bond = torch.reshape(distances, (m, k))
            atom_nbr_fea = torch.Tensor(
                np.array([[pos[j] for j in _nbr_idx[i]] for i in range(m)])
            )
            centre_coords = pos.unsqueeze(1).expand(m, k, 3)
            dxyz = atom_nbr_fea - centre_coords
            r = bond.unsqueeze(2)
            angle_cos = torch.matmul(dxyz, torch.swapaxes(dxyz, 1, 2)) / torch.matmul(
                r, torch.swapaxes(r, 1, 2)
            )
            angle_cos = angle_cos.flatten(0, 1)  # To fit into collate

            rattled_graphs.append(
                Data(
                    num_nodes=graph.num_nodes,
                    pos=pos,
                    cell=graph.cell,
                    edge_index=graph.edge_index,
                    edge_dist=distances,
                    angle_cos=angle_cos,
                )
            )

            # yield Data(
            #     num_nodes=graph.num_nodes,
            #     pos=pos,
            #     cell=graph.cell,
            #     edge_index=graph.edge_index,
            #     edge_dist=distances,
            #     angle_cos=angle_cos,
            # )

            if progress_bar:
                pbar.update(1)

        if progress_bar:
            pbar.close()

        return rattled_graphs

    @staticmethod
    def _pbc_preparation(cell: torch.Tensor) -> torch.Tensor:
        """Returns all the translations to apply to positions in order to
        get images from periodic images of cells adjacent to the main one.

        Args:
            cell (torch.Tensor): 2D tensor with all 3 lattice vectors
        Returns:
            trans (torch.Tensor): 2D tensor with all translations to apply to positions
                                  in order to get periodic images.
        """
        latvec_multiplier = torch.FloatTensor(
            [m for m in itertools.product(torch.arange(3) - 1, repeat=3)]
        )
        return torch.sum(
            torch.transpose(
                torch.transpose(cell.unsqueeze(0).repeat(27, 1, 1), dim0=1, dim1=2)
                * latvec_multiplier[:, None],
                dim0=1,
                dim1=2,
            ),
            dim=1,
        )

    @staticmethod
    def _distance(
        atom_pos: torch.Tensor, neigh_pos: torch.Tensor, trans: torch.Tensor
    ) -> torch.Tensor:
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
        ne_trans = neigh_pos.unsqueeze(0).repeat(27, 1) + trans
        return torch.nn.functional.pairwise_distance(atom_pos * torch.ones([27, 3]), ne_trans)


# TODO set as deprecated for now and change it if we start using the positions and cell
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
        p (float): probability to apply the transform (default=0.1).
    """

    def __init__(self, stddev: float = 0.05, seed: int = 42, p: float = 0.1) -> None:
        super().__init__()
        self.stddev = stddev
        self.p = p
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def forward(self, x: Data) -> Data:
        """Applies random expansion to a batch of graphs."""
        if torch.rand(1, generator=self.rng).item() > self.p:
            return x

        scaled_pos = None
        if x.pos is not None:
            rnd_scale = torch.normal(mean=1.0, std=self.stddev, size=[1], generator=self.rng)
            scaled_pos = x.pos * rnd_scale

        scaled_cell = None
        if x.cell is not None:
            rnd_scale = torch.normal(mean=1.0, std=self.stddev, size=[1], generator=self.rng)
            scaled_cell = x.cell * rnd_scale

        # ? GH I think it is the same as using RandomDisplacement
        scaled_dist = None
        if x.edge_dist is not None:
            rnd_scale = torch.normal(mean=1.0, std=self.stddev, size=[1], generator=self.rng)
            scaled_dist = x.edge_dist * rnd_scale

        augmented_data = Data(
            num_nodes=x.num_nodes,
            pos=scaled_pos,
            cell=scaled_cell,
            edge_index=x.edge_index,
            edge_dist=scaled_dist,
            angle_cos=x.angle_cos,  # unchanged by box scaling
        )

        return augmented_data


# FIXME: this class is not used in the current implementation
# as it will make the code crash. We need to move to the usual PyG
# Data storage of the targets and use the PyG collate function.
class RandomNodeDrop(torch.nn.Module):
    """Class to apply random node dropout to boxes.

    Randomly drops every node of a graph with probability p.
    This mimics crystalline defects in units cells (without
    relaxation).

    Args:
        rate (float): dropout probability in [0.,1.] (default=0.05).
        seed (int): Random seed (default=42).
        p (float): probability to apply the transform (default=0.1).
    """

    def __init__(self, rate: float = 0.05, seed: int = 42, p: float = 0.1) -> None:
        if rate == 1.0:
            raise ValueError("The dropout rate must be strictly less than 1.0.")

        if p < 0.0 or p > 1.0:
            raise ValueError("The dropout probability must be in [0.,1.].")

        super().__init__()

        self.rate = rate
        self.p = p
        self.seed = seed
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def forward(self, x: Data) -> Data:
        """Applies random node dropout to a batch of graphs."""
        # TODO we may switch to modify the Data object every time but with a given probability
        if torch.rand(1, generator=self.rng).item() > self.p:
            return x

        if x.num_nodes is None:
            raise ValueError("The number of nodes must be provided.")

        num_nodes_kept = 0
        while num_nodes_kept == 0:
            node_mask = torch.rand(x.num_nodes, generator=self.rng) > self.rate
            num_nodes_kept = torch.sum(node_mask).item()

        # Updated nodes/positions
        if x.pos is not None:
            pos = x.pos[node_mask]

        # Need to remove edges involving the dropped nodes
        if x.edge_index is not None:
            edge_mask = node_mask[x.edge_index[0]]
            edge_index = x.edge_index[:, edge_mask]

        edge_dist = x.edge_dist[edge_mask] if edge_mask is not None else None
        angle_cos = x.angle_cos[edge_mask] if edge_mask is not None else None

        augmented_graph = Data(
            num_nodes=num_nodes_kept,
            pos=pos,
            cell=x.cell,
            edge_index=edge_index,
            edge_dist=edge_dist,
            angle_cos=angle_cos,
        )

        return augmented_graph
