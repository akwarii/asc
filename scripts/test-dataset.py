from src.datasets import CSG, Aflow, CustomDataset, MaterialProject

k = 12
rcut = 6.0
reload = True

custom = CustomDataset(force_reload=reload, k=k, rcut=rcut)
print("Num classes: ", custom.num_classes)
custom.print_summary()

aflow = Aflow(force_reload=reload, k=k, rcut=rcut)
print("Num classes: ", aflow.num_classes)
aflow.print_summary()

csg = CSG(force_reload=reload, k=k, rcut=rcut)
print("Num classes: ", csg.num_classes)
csg.print_summary()

mp = MaterialProject(force_reload=reload, k=k, rcut=rcut)
print("Num classes: ", mp.num_classes)
mp.print_summary()
