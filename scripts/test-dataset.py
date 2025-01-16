from src.datasets import CSG, Aflow, CustomDataset, MaterialProject

k = 12
rcut = 6.0
reload = True

custom = CustomDataset(force_reload=reload, k=k, rcut=rcut)
custom.print_summary()

csg = Aflow(force_reload=reload, k=k, rcut=rcut)
csg.print_summary()

csg = CSG(force_reload=reload, k=k, rcut=rcut)
csg.print_summary()

mp = MaterialProject(force_reload=reload, k=k, rcut=rcut)
mp.print_summary()
