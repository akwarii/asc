from src.datasets import CSG
<<<<<<< HEAD
=======
from src.transforms.line_graph import LineGraph
>>>>>>> b9409e20c20ee718a93ab1d169351211166c6659

rcut = 6.0
reload = True

for k in [10, 12, 14, 16]:
<<<<<<< HEAD
    dataset = CSG(force_reload=reload, k=k, rcut=rcut)
=======
    dataset = CSG(
        pre_transform=LineGraph(),
        force_reload=reload,
        k=k,
        rcut=rcut,
    )
>>>>>>> b9409e20c20ee718a93ab1d169351211166c6659
    print(dataset)
    print("Num classes: ", dataset.num_classes)
    dataset.print_summary()
