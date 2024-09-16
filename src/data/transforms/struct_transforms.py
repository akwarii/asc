# TODO add tags to atoms in the pymatgen structure
# if the tag is 0, then the atom should not be included in the graph
# if the tag is 1, then the atom should be included in the graph
# if the tag is None, then the atom should be included in the graph and the tag is not used

# DB : pymatgen Structures include built-in labels for each sites
#      -> check if the codes already uses it, else take advantage of those
# else, look at on-site properties ?
# Eventually, we can try altering the actual Structure object to add tags.
# https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure