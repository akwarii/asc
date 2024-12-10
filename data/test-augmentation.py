from src.data.datasets.material_project import MaterialProject
from src.processing.graph import KNNGraph
from src.data.augmentation.augmentation import RandomDisplacement, RandomExpansion, RandomNodeDrop
from pprint import pp
import numpy as np

from tqdm import tqdm

import torch

# Source data
# mp = MaterialProject(root="data/mp-data", download=False)
# # Computing graphs
# graph_builder = KNNGraph()
# graphs = []
# pbar = tqdm(total=len(mp.data), desc="Computing original graphs.")
# for data in mp.data :
#     struct = graph_builder._to_pymatgen_struct(data)
#     graphs.append(graph_builder.convert(struct=struct))
#     pbar.update(1)
# pbar.close()
# torch.save(graphs, "./data/saved-mp-graphs.pt")

# graph_converter = KNNGraph()
# structs = []
# for data in mp.data :
#     structs.append(graph_converter._to_pymatgen_struct(data))
# batch = graph_converter.batch_conversion(structs=structs)

batch = torch.load("./data/saved-mp-graphs.pt")
# # augmented_graphs = torch.load("./data/saved-mp-augmented-rattle-graphs.pt")
# # augmented_graphs = torch.load("./data/saved-mp-augmented-expansion-graphs.pt")
# # augmented_graphs = torch.load("./data/saved-mp-augmented-dropout-graphs.pt")

# # Augmenting the graphs
augmenter = RandomDisplacement()
# augmenter = RandomExpansion()
# augmenter = RandomNodeDrop()
augmented_graphs = augmenter.forward(x=batch)


torch.save(augmented_graphs, "./data/saved-mp-augmented-rattle-graphs.pt")
# torch.save(augmented_graphs, "./data/saved-mp-augmented-expansion-graphs.pt")
# torch.save(augmented_graphs, "./data/saved-mp-augmented-dropout-graphs.pt")
