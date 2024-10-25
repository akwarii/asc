import torch
import numpy as np
from pymatgen.core import Structure
from collections.abc import Sequence
# TODO add tags to atoms in the pymatgen structure
# if the tag is 0, then the atom should not be included in the graph
# if the tag is 1, then the atom should be included in the graph
# if the tag is None, then the atom should be included in the graph and the tag is not used

# DB : Per data/cegann_datamodule, struct_transorms are to be applied before graph
# creation. In ths regard, it might be suitable to simply remove the atoms from
# the structure rather.
class RemoveAtoms(torch.nn.Module) :
    """Removes atoms in a pymatgen Structure.
    If the tag is zero, the atom should not be included in the graph.
    If the tag is non-zero, the atom should be included in the graph.
    If the tag is None, the atom should be included in  the graph and
    the tag is not used.

    Args:
        None
    
    Methods:
        forward: performs the removal of atoms in the structure
        _label_sites: given either atom index or species, applies labels to atoms in the structure.
    """

    def __init__(
            self
    ) -> None :
        super.__init__()

    def forward(
        self,
        structs: Sequence[Structure] = [],
        species: Sequence[Sequence[str]] = [],
        masks: Sequence[Sequence[int]] = [],
        indexes: Sequence[Sequence[int]] = []
    ) -> None :
        """Modifies a batch of pymatgen Sequences to remove atomic sites based
        on their internal `site_properties` parameter, under the `"keep-site?"`
        key. Also modifies those `site_properties` beforehand in the case instructions are provided.

        Args:
            structs (Sequence[pymatgen.core.Structure]) : batch of pymatgen Structures in which site removal has to be performed.
            species (Sequence[Sequence[str]]) : list of chemical species to remove for each Structure.
            masks (Sequence[Sequence[int]]) : list of masks to apply over each Structure individual site (0=remove atom, 1=keep it).
            indexes (Sequence[Sequence[int]]) : list of individual sites to remove from each Structure.
        """

        ######################################################################
        #                              NOTE                                  #
        ######################################################################
        # In theory, Pymatgen Structures already embedd routines in order to 
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
        # rameter dictionnary, with the `"keep-site?"` key.
        ######################################################################

        # (1) https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure.remove_species
        # (2) https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure.remove_sites

        # Change labels on structures, if some instructions are present
        args = {}
        if len(species) == len(structs) : args["species"] = species
        if len(masks)   == len(structs) : args["masks"]   = masks
        if len(indexes) == len(structs) : args["indexes"] = indexes
        for struct in structs :

            self._label_sites(struct=structs, **args)

            struct.remove_sites(
                np.where(
                    np.logical_not(struct.site_properties["keep-site?"])
                )
            )
        


    @staticmethod
    def _label_sites(
        struct: Structure,
        species: Sequence[str] = [],
        indexes: Sequence[int] = [],
        mask: Sequence[int] = []
    ) -> None :
        """Applies removal instructions on a pymatgen Structure. Those are stored
        in the `site_properties` parameter dictionnary, under the `"keep-site?" key,
        given species and/or a lists of integers either being the indexes to remove
        and/or a mask with zeros and ones, zeros meaning the atoms to remove from
        the Structure.

        Args:
            struct (pymatgen.core.Structure): the Structure on which the labels will be applied.
            species (Seqence[str]) : the chemical species to remove from struct.
            indexes (Sequence[int]) : indexes of atoms to remove from struct.
            mask (Sequence[int]) : mask of [0,1] to select sites to be removed/kept.

        """
        n_atoms = np.size(struct.species)
        labels = np.ones(n_atoms)
        
        # mask with 0 and 1 for each site
        assert len(mask) == n_atoms 
        labels *= mask
        
        # indexes of atoms to delete from the structure
        assert len(indexes) <= n_atoms
        labels[indexes] = 0

        # Using provided species
        assert len(species) <= n_atoms
        labels *= [
            not s in species for s in np.array(struct.species).astype(str)
        ]

        struct.add_site_property("keep-site?", labels.astype(bool))

