from collections.abc import Callable
from typing import Any

from torch_geometric.data import InMemoryDataset


class CSG(InMemoryDataset):
    """The Crystal Space Group (CSG) dataset is a preprocessed version of the AFLOW, GNoME, and
    Material Project datasets. The dataset contains ~470,000 crystal structures with space group
    numbers ranging from 1 to 230. The dataset is formatted as a CSV file with two columns:
    "SpaceGroupNumber" and "Structure". The "Structure" column contains the string representation
    of the crystal structure in the POSCAR format.

    AFLOW data has been filtered to include only structures with a maximum stress component of
    +/-0.1 GPa and a maximum force component of +/-0.01 eV/A. Material Project data was filtered
    to remove structures with depreciation or warning flags. All GNoME data predicted to be stable
    were included. In addition, structures with the same space group number and composition were
    removed to avoid redundancy. To reduce class imbalance to ~100, the most represented classes
    are limited to 10,000 samples, while the least represented have a minimum of 100 samples.The
    maximum number of atoms in a structure is 444.

    The dataset is available for download from Kaggle at https://www.kaggle.com/datasets/gaelhuynh/space-group.

    Args:
        root: Root directory of the dataset.
        transform: A function that takes in a graph and returns a transformed version.
        pre_transform: A function that takes in a graph and returns a transformed version.
        pre_filter: A function that takes in a graph and returns a boolean value indicating
            whether the graph should be included in the dataset.
        force_reload: Whether to reload the dataset even if it already exists.
        kwargs: Additional keyword arguments to be passed to the KNNGraph or InMemoryDataset class.

    Attributes:
        KAGGLE_DATASET (str): The name of the Kaggle dataset.
    """

    KAGGLE_DATASET = "gaelhuynh/space-group-small"

    def __init__(
        self,
        root: str = "data/csg",
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
        **kwargs: Any,
    ) -> None:
        self.kwargs = kwargs.copy()

        kwargs.pop("k", None)
        kwargs.pop("rcut", None)
        super().__init__(
            root, transform, pre_transform, pre_filter, force_reload=force_reload, **kwargs
        )

        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        """Return the name of the downloaded files."""
        return ["CSG_small.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the name of the processed files ie the transformed data saved to the disk."""
        return ["data.pt"]

    def download(self) -> None:
        """Download the dataset from Kaggle and store it in the raw directory."""
        from dotenv import load_dotenv

        try:
            from kaggle import KaggleApi
        except ImportError:
            raise ImportError(
                "The Kaggle API client is not installed. Install it with `pip install kaggle`."
            )

        load_dotenv()
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(self.KAGGLE_DATASET, path=self.raw_dir, quiet=False, unzip=True)

    def process(self) -> None:
        """Process the dataset by converting the structures to graphs, applying both pre-filter and
        pre-transform functions, and saving the processed data to disk. The data is saved in the
        processed directory as a single file named "data.pt".
        """
        import warnings

        import pandas as pd
        import torch
        from pymatgen.io.vasp.inputs import BadPoscarWarning
        from tqdm.auto import tqdm

        from src.graph import KNNGraph

        df = pd.read_csv(self.raw_paths[0])
        knn = KNNGraph(**self.kwargs)

        # Convert the target labels to consecutive 0-based indices
        unique_labels = sorted(set(df["SpaceGroupNumber"]))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        df["SpaceGroupNumber"] = df["SpaceGroupNumber"].map(label_to_index)

        data_list = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BadPoscarWarning)
                data = knn.convert(row["Structure"])

            if data.num_nodes is None or data.num_nodes == 0:
                raise RuntimeError("The number of nodes in the graph is zero.")

            data.y = torch.full((data.num_nodes,), int(row["SpaceGroupNumber"]))

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
