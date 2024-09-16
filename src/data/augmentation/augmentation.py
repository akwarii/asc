import torch


# TODO implement the gaussian noise augmentation
# DB : https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.rattle
class RandomDisplacement(torch.nn.Module):
    pass


# TODO implement the random expansion augmentation
class RandomExpansion(torch.nn.Module):
    pass

# TODO implement molecular dynamics data augmentation
# DB : https://wiki.fysik.dtu.dk/ase/ase/md.html#module-ase.md
class MolecularDynamics(torch.nn.Module):
    pass
