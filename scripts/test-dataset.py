from src.datasets import CSG, MaterialProject

csg = CSG(force_reload=True)
print(csg.get_summary())

mp = MaterialProject(force_reload=True)
print(mp.get_summary())
