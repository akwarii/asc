REPR_INDENT = 4

_ALL_SG = set(range(1, 231))

# DB
_NOT_IN_AFLOW = {
    # 16,  # DB, found in the dataset
    # 17,  # DB, found in the dataset
    # 24,  # DB, found in the dataset
    # 27,  # DB, found in the dataset
    # 30,  # DB, found in the dataset
    # 32,  # DB, found in the dataset
    # 34,  # DB, found in the dataset
    # 39,  # DB, found in the dataset
    # 41,  # DB, found in the dataset
    # 45,  # DB, found in the dataset
    # 48,  # DB, found in the dataset
    49,
    # 50,  # DB, found in the dataset
    # 54,  # DB, found in the dataset
    # 73,  # DB, found in the dataset
    # 77,  # DB, found in the dataset
    # 78,  # DB, found in the dataset
    # 79,  # DB, found in the dataset
    # 80,  # DB, found in the dataset
    89,
    # 90,  # DB, found in the dataset
    93,
    94,
    # 95,  # DB, found in the dataset
    # 97,  # DB, found in the dataset
    # 98,  # DB, found in the dataset
    101,
    # 103, # DB, found in the dataset
    # 104, # DB, found in the dataset
    # 106, # DB, found in the dataset
    # 112, # DB, found in the dataset
    # 118, # DB, found in the dataset
    # 143, # DB, found in the dataset
    # 144, # DB, found in the dataset
    # 145, # DB, found in the dataset
    # 158, # DB, found in the dataset
    168,
    # 169, # DB, found in the dataset
    # 170, # DB, found in the dataset
    171,
    172,
    # 177, # DB, found in the dataset
    # 179, # DB, found in the dataset
    # 184, # DB, found in the dataset
    # 195, # DB, found in the dataset
    # 196, # DB, found in the dataset
    207,
    209,
    # 210, # DB, found in the dataset
    # 211, # DB, found in the dataset
    # 222, # DB, found in the dataset
    228,
}
AFLOW_CLASSES = tuple(sorted(_ALL_SG - _NOT_IN_AFLOW))

_NOT_IN_MP = {168, 207}
MP_CLASSES = tuple(_ALL_SG - _NOT_IN_MP)

_NOT_IN_GNOME = {
    # 48, # DB, found in the dataset
    78,
    93,
    96,
    106,
    145,  # DB, not found in the dataset
    153,
    158,
    168,
    169,
    170,
    171,
    172,
    177,  # DB, not found in the dataset
    178,
    179,
    181,  # DB, not found in the dataset
    184,
    195,
    201,  # DB, not found in the dataset
    207,
    208,
    210,
    211,
    219,
    222,
    228,
}
GNOME_CLASSES = tuple(_ALL_SG - _NOT_IN_GNOME)

_NOT_IN_CSG = {168, 207}
CSG_CLASSES = tuple(_ALL_SG - _NOT_IN_CSG)

CUSTOM_CLASSES = tuple(set(range(8)))  # DB #TODO this should be found in the dataset
