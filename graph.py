import sys
import dgl
import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from src.utils.bonds import compute_bond_cosines
from src.utils.neighbors import find_knn_in_shell

# source = [0, 1, 2, 3, 4, 5]
# destination = [1, 2, 3, 4, 5, 6]
# graph = dgl.graph((source, destination))
# print(graph.adj_external().to_dense())

# struct = Structure.from_file('example/train/0.POSCAR')
# struct_graph = StructureGraph.with_empty_graph(struct)

class Graph:
    def __init__(self, neighbors: int = 12, rcut: float = 0, delta: float = 1, self_loop: bool = True) -> None:
        self.n_neighbors = neighbors
        self.rcut = rcut
        self.delta = delta
        self.self_loop = self_loop

    def set_features(self, structure: Structure) -> None:
        """
        Generate the graph from a pymatgen structure. Note that the graph is
        constructed from the whole structure and the neighbor list is just a
        way to efficiently grep the subgraph of each node.
        """
        all_neighbors_sorted = find_knn_in_shell(structure, self.rcut, self.n_neighbors, self.delta)
        
        # Create U, V arrays for the graph
        u, v = [], []
        for i, neighbors in enumerate(all_neighbors_sorted):
            for neighbor in neighbors:
                u.append(i)
                v.append(neighbor.index)
                
        self.g = dgl.graph((u, v), num_nodes=len(structure.sites), idtype=torch.int32)

        # Set node and edge features
        self.g.ndata["coords"] = torch.from_numpy(structure.cart_coords).float()
        #FIXME: shape is (n_nodes, 1) instead of (n_nodes, n_neighbors)
        self.g.ndata["neighbors"] = torch.from_numpy(np.array([x[2] for neighbors in all_neighbors_sorted for x in neighbors], dtype=np.int32))
        #FIXME: shape is (n_edges, 1) instead of (n_edges, n_neighbors)
        self.g.edata["r"] = torch.from_numpy(np.array([x[1] for neighbors in all_neighbors_sorted for x in neighbors], dtype=np.float32))

        # Add self loops
        if self.self_loop:        
            self.g.add_edges(self.g.nodes(), self.g.nodes())
            
        self.lg = dgl.line_graph(self.g, shared=True)
        self.lg.apply_edges(compute_bond_cosines)


if __name__ == "__main__":
    torch.set_printoptions(profile="full", linewidth=500)
    struct = Structure.from_file('example/train/0.POSCAR')
    graph = Graph()
    graph.set_features(struct)
    print(graph.g.num_nodes())
    print(graph.g.num_edges())
    # neighbor list of node 0 (note that node 0 is included because of the self loop)
    print(graph.g)
    print(graph.lg)