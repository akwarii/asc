from src.datasets import MaterialProject
from src.transforms.struct_transforms import RemoveAtoms
from src.graph import KNNGraph
import numpy as np

from tqdm import tqdm

# Source data
mp = MaterialProject(root="data/mp-data", fetch_data=False)

# Structures
graph_converter = KNNGraph()
structs = []
pbar = tqdm(total=len(mp.data), desc="Converting to structures.")
for data in mp.data:
    structs.append(graph_converter._to_pymatgen_struct(data))
    pbar.update(1)
pbar.close()

# Removing atoms
transformer = RemoveAtoms()
indexes_to_remove = []

avg_init = 0
for struct in structs:
    indexes_to_remove.append([0])
    avg_init += np.size(struct.species)
avg_init = float(avg_init) / float(len(structs))

transformer.forward(structs=structs, indexes=indexes_to_remove)
avg_final = 0
for struct in structs:
    avg_final += np.size(struct.species)
avg_final = float(avg_final) / float(len(structs))

print("Average numer of atoms before and after removal")
print("   (difference should be one)")
print(avg_init, avg_final)
