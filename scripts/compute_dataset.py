from src.datasets import CSG
from src.transforms.line_graph import LineGraph

rcut = 6.0
reload = True

for k in [10, 12, 14, 16]:
    dataset = CSG(
        pre_transform=LineGraph(),
        force_reload=reload,
        k=k,
        rcut=rcut,
    )
    print(dataset)
    print("Num classes: ", dataset.num_classes)
    dataset.print_summary()
