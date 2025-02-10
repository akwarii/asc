from itertools import product

from src.datasets import CSG, Aflow, CustomDataset, MaterialProject

k = 12
rcut = 6.0
reload = True

datasets = [CustomDataset, Aflow, CSG, MaterialProject]
knn = [10, 12, 14, 16]

for dataset, k in product(datasets, knn):
    data = dataset(force_reload=reload, k=k, rcut=rcut)
    print(dataset)
    print("Num classes: ", data.num_classes)
    data.print_summary()
