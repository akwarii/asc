import sys

import torch
from tqdm.auto import tqdm

from src.augmentation import (
    RandomDisplacement,
    RandomExpansion,
    RandomNodeDrop,
)
from src.datasets import MaterialProject
from src.graph import KNNGraph

mp = MaterialProject(root="data")
# graphs = KNNGraph().convert_and_save(mp.data[:1_000], mp.processed_folder)
# sys.exit()
knn = KNNGraph()
# graphs = []
# for struct in tqdm(mp.data[:1_000]):
#     graph = knn.convert(struct)
#     graphs.append(graph)

batch = list(knn.batch_conversion(structs=mp.data[:10_000]))
# batch = torch.load("./data/saved-mp-graphs.pt")
# # augmented_graphs = torch.load("./data/saved-mp-augmented-rattle-graphs.pt")
# # augmented_graphs = torch.load("./data/saved-mp-augmented-expansion-graphs.pt")
# # augmented_graphs = torch.load("./data/saved-mp-augmented-dropout-graphs.pt")

# # Augmenting the graphs
# augmenter = RandomDisplacement()
# augmenter = RandomExpansion()
augmenter = RandomNodeDrop()
augmented_graphs = augmenter.forward(x=batch)


# torch.save(augmented_graphs, "./data/saved-mp-augmented-rattle-graphs.pt")
# torch.save(augmented_graphs, "./data/saved-mp-augmented-expansion-graphs.pt")
# torch.save(augmented_graphs, "./data/saved-mp-augmented-dropout-graphs.pt")
