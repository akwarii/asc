from collections.abc import Callable
from typing import Any

import pandas as pd
import torch
from dotenv import load_dotenv
from kaggle import KaggleApi
from pymatgen.core import Structure
from torch_geometric.data import InMemoryDataset

from src.graph import KNNGraph


class CSG(InMemoryDataset):
    """The Crystal Space Group (CSG) dataset is a preprocessed version of the AFLOW, GNoME and
    Material Project datasets. The dataset contains ~1,050,000 crystal structures with space group
    numbers ranging from 1 to 230. The dataset is formatted as a CSV file with two columns:
    "SpaceGroupNumber" and "Structure". The "Structure" column contains the string representation
    of the crystal structure in the POSCAR format.

    The AFLOW data was filtered to only include structures with a maximum stress component of +/-
    0.1 GPa and a maximum force component of +/-0.01 eV/A. Material Project data was filtered to
    remove structures with deprecated or warning flags. All GNoME data predicted stable were
    included. Additionally, structures with both the same space group number and composition were
    removed to avoid redundancy.

    The maximum number of atoms in a structure is 444. A radius graph with a cutoff of 10 angstroms
    ensure that 99.5% of the graphs are complete.

    The dataset is available for download from Kaggle at https://www.kaggle.com/datasets/gaelhuynh/space-group.

    Args:


    Attributes:

    """

    KAGGLE_DATASET = "gaelhuynh/space-group"

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(root, transform, pre_transform, pre_filter)

        self.kwargs = kwargs
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        """Return the name of the downloaded files."""
        return ["CSG.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the name of the processed files ie the transformed data saved to the disk."""
        return ["data.pt"]

    def download(self) -> None:
        """Downloads the dataset from Kaggle if it doesn't exist already or if its md5
        checksum doesn't match the expected value.
        """
        load_dotenv()

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.KAGGLE_DATASET, path=self.raw_dir, quiet=False, unzip=True)

    def process(self) -> None:
        df = pd.read_csv(self.raw_paths[0])
        knn = KNNGraph(**self.kwargs)

        data_list = []
        for _, row in df.iterrows():
            struct = Structure.from_str(row["Structure"], fmt="poscar")
            data = knn.convert(struct)
            data.y = torch.full((len(struct), ), int(row["SpaceGroupNumber"]))

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
