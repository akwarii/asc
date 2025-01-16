from collections.abc import Sequence

import numpy as np
import torch
from pymatgen.core import Structure

# TODO add tags to atoms in the pymatgen structure
# if the tag is 0, then the atom should not be included in the graph
# if the tag is 1, then the atom should be included in the graph
# if the tag is None, then the atom should be included in the graph and the tag is not used


# DB : Per data/cegann_datamodule, struct_transorms are to be applied before graph
# creation. In this regard, it might be suitable to simply remove the atoms from
# the structure rather.
class RemoveAtoms(torch.nn.Module):
    """Removes atoms in a pymatgen Structure. If the tag is zero, the atom should not be included
    in the graph. If the tag is non-zero, the atom should be included in the graph. If the tag is
    None, the atom should be included in  the graph and the tag is not used.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        structs: Sequence[Structure] = [],
        species: Sequence[Sequence[str]] = [],
        masks: Sequence[Sequence[int]] = [],
        indexes: Sequence[Sequence[int]] = [],
        progress_bar: bool = True,
    ) -> None:
        """Modify a batch of pymatgen Sequences to remove atomic sites based on their internal
        `site_properties` parameter, under the `keep-site?` key. Also modifies those
        `site_properties` beforehand in the case instructions are provided.

        Args:
            structs: batch of pymatgen Structures in which site removal has to be performed
            species: list of chemical species to remove for each Structure
            masks: list of masks to apply over each individual site (0=remove atom, 1=keep it)
            indexes: list of individual sites to remove from each Structure
            progress_bar: whether we show progression using tqdm

        Returns:
            None
        """
        ######################################################################
        #                              NOTE                                  #
        ######################################################################
        # In theory, Pymatgen Structures already embed routines in order to
        # remove atoms based on `species` - namely `remove_species` - and on
        # `indexes` - namely `remove_sites`. See the links (1) and (2) below
        # for more details.
        #
        # This means we could be able to use them directly for sites removal
        # given `species` and `indexes`, but it would lead to two issues :
        #  * we could then not use `masks`
        #  * the user could then not embed their own labels in the Structure
        #
        # Hence, I prefer to embed those labels in the `site_properties` pa-
        # rameter dictionary, with the `"keep-site?"` key.
        ######################################################################

        # (1) https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure.remove_species
        # (2) https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure.remove_sites

        # Change labels on structures, if some instructions are present

        use_species = len(structs) == len(species)
        use_masks = len(structs) == len(masks)
        use_indexes = len(structs) == len(indexes)

        if progress_bar:
            from tqdm import tqdm

            pbar = tqdm(total=len(structs), desc="Removing unwanted sites from structures.")

        species_mask = None
        indexes_mask = None
        sites_mask = None
        for i, struct in enumerate(structs):
            if use_species:
                species_mask = species[i]
            if use_indexes:
                indexes_mask = indexes[i]
            if use_masks:
                sites_mask = masks[i]

            self._label_sites(
                struct=struct,
                species=species_mask,
                indexes=indexes_mask,
                mask=sites_mask,
            )

            to_remove = np.where(np.logical_not(struct.site_properties["keep-site?"]))[0]
            struct.remove_sites(to_remove.tolist())

            if progress_bar:
                pbar.update(1)

        if progress_bar:
            pbar.close()

    @staticmethod
    def _label_sites(
        struct: Structure,
        species: Sequence[str] | None = None,
        indexes: Sequence[int] | None = None,
        mask: Sequence[int] | None = None,
    ) -> None:
        """Apply removal instructions on a pymatgen Structure. Those are stored in the
        `site_properties` parameter dictionary, under the `"keep-site?" key, given species and/or
        a lists of integers either being the indexes to remove and/or a mask with zeros and ones,
        zeros meaning the atoms to remove from the Structure.

        Args:
            struct: the Structure on which the labels will be applied.
            species: the chemical species to remove from struct.
            indexes: indexes of atoms to remove from struct.
            mask: mask of [0,1] to select sites to be removed/kept.

        """
        n_atoms = np.size(struct.species)
        labels = np.ones(n_atoms)

        # mask with 0 and 1 for each site
        if mask is not None:
            assert len(mask) == n_atoms
            labels *= mask

        # indexes of atoms to delete from the structure
        if indexes is not None:
            assert len(indexes) <= n_atoms
            labels[indexes] = 0

        # Using provided species
        if species is not None:
            assert len(species) <= n_atoms
            labels *= [s not in species for s in np.array(struct.species).astype(str)]

        # Storing sites to keep as booleans
        struct.add_site_property("keep-site?", labels.astype(bool))
