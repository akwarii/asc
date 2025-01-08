from src.datasets import CSG, Aflow, CustomDataset, MaterialProject

reload = False

custom = CustomDataset(force_reload=reload, k=8, rcut=5.0)
print(custom.get_summary())

csg = Aflow(force_reload=reload, k=8, rcut=5.0)
print(csg.get_summary())

csg = CSG(force_reload=reload, k=8, rcut=5.0)
print(csg.get_summary())

mp = MaterialProject(force_reload=reload, k=8, rcut=5.0)
print(mp.get_summary())
