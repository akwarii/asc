# Lightning-CEGANN2
A Lightning implementation of a modified Crystal Edge Graph Attention Neural Network

Modifications include:
- [X] Cleanning the code
- [ ] Creation of a CLI (Argparse)
- [ ] Add a better logging system (Logging)
- [ ] Use of Lightning instead of Ignite (Ignite -> Lightning)
- [X] Change attention mechanism to GATv2 (without PyG)
- [ ] Implementation of hyperparameter optimization and pruning (Optuna)
- [ ] Optimization of the graph creation
- [ ] Database creation from either Gaussian noise or MD (ASE/Pymatgen)

Experiments:
- [ ] Local classification of space groups
- [ ] Local classification of space groups and grain boundaries

Guideline:
1. Start with the space groups needed for zirconia
2. Add the grain boundaries to the DB
3. Try to classify the 230 space groups (Starting from the well converged structures in Material project)
4. If it works well, try to add grain boundaries
