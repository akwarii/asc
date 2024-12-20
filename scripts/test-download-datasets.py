import json
import pprint

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from src.datamodule import CEGANNDataModule
from src.datasets import MaterialProject

# mp = MaterialProject(root="./data/mp-data", fetch_data=True)
mp = MaterialProject(root="data/mp-data", fetch_data=False)

rattled_atoms = []
for i, data in enumerate(mp.data[:3]):
    at0 = Structure.from_str(data, fmt="poscar").to_ase_atoms()
    at1 = at0.copy()
    at1.rattle()  # default stdev intensity ok ?
    st1 = AseAtomsAdaptor.get_structure(at1).to(fmt="poscar")

    rattled_atoms.append({"spacegroup": mp.targets[i], "structure": st1})

with open("data/rattled_structures.json", "w") as f:
    json.dump(rattled_atoms, f, indent=4)

# @GAEL DO WE WANT DATA AUGMENTATION WITHIN CEGANN DATAMODULE
# OR DO WE PERFORM IT BEFOREHAND (dedicated command) AND SAVE
# IT IN A DEDICATED DIRECTORY ?

datamodule = CEGANNDataModule(root="data/mp-data", transforms=None, datasets="mp")

datamodule.prepare_data()

datamodule.setup(stage="fit")

traincdm = datamodule.train_dataloader()
testcdm = datamodule.test_dataloader()
valcdm = datamodule.val_dataloader()
predcdm = datamodule.predict_dataloader()

pprint.pp(traincdm.dataset)  # <torch.utils.data.dataset.Subset object at xx>
pprint.pp(traincdm.dataset.__getitem__(0))  # (<src.processing.graph.KNNGraph object at xx>, 194)
pprint.pp(traincdm.dataset.__getitem__(1))  # (<src.processing.graph.KNNGraph object at xx>, 2)
pprint.pp(traincdm.dataset.__getitem__(2))  # (<src.processing.graph.KNNGraph object at xx>, 69)

pprint.pp(traincdm.dataset.__getitem__(0)[0])  # <src.processing.graph.KNNGraph object at xx>
pprint.pp(traincdm.dataset.__getitem__(0)[1])  # 194
if datamodule.data_train is not None:
    pprint.pp(datamodule.data_train[0])  # like traincdm.dataset.__getitem__(0)

pprint.pp(datamodule.hparams.datasets[0])  # 'mp'

# tst = Structure.from_str("wrongString", fmt="poscar")

# cdm.on_before_batch_transfer(traincdm.dataset, 0)

# pprint.pp(traincdm.dataset)

# pprint.pp(traincdm.dataset[0])
