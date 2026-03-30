import os.path as osp
from collections.abc import Callable
from typing import Any

from src.datasets.base import Dataset


class CSG(Dataset):
    """The Crystal Space Group (CSG) dataset is a preprocessed version of the AFLOW, GNoME, and
    Material Project datasets. The dataset contains ~200,000 crystal structures with space group
    numbers ranging from 1 to 230. The dataset is formatted as a CSV file with two columns:
    "SpaceGroupNumber" and "Structure". The "Structure" column contains the string representation
    of the crystal structure in the POSCAR format.

    AFLOW data has been filtered to include only structures with a maximum stress component of
    +/-0.1 GPa and a maximum force component of +/-0.01 eV/A. Material Project data was filtered
    to remove structures with depreciation or warning flags. All GNoME data predicted to be stable
    were included. In addition, structures with the same space group number and composition were
    removed to avoid redundancy. To reduce class imbalance to ~25, the most represented classes
    are limited to 2,500 samples, while the least represented have a minimum of 100 samples. The
    maximum number of atoms is limited to 50. In the end, 149 space groups are represented in the
    dataset.

    The dataset is available for download from Kaggle at
    https://www.kaggle.com/datasets/gaelhuynh/space-group-small.

    Args:
        root: Root directory of the dataset.
        transform: A function that takes in a graph and returns a transformed version.
        pre_transform: A function that takes in a graph and returns a transformed version.
        pre_filter: A function that takes in a graph and returns a boolean value indicating
            whether the graph should be included in the dataset.
        force_reload: Whether to reload the dataset even if it already exists.
        download_only: Whether to download the dataset only without processing and loading it.
        kwargs: Additional keyword arguments to be passed to PeriodicKNN or Dataset class.

    Attributes:
        KAGGLE_DATASET (str): The name of the Kaggle dataset.
    """

    KAGGLE_DATASET = "gaelhuynh/space-group-tiny"

    def __init__(
        self,
        root: str = "data/csg",
        *,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
        force_reload: bool = False,
        download_only: bool = False,
        search_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        # TODO: Implement search_kwargs support for CSG dataset queries.
        # ? eg. this could be an easy way to swith between "CSG.csv" and "CSG_tiny.csv"
        self.search_kwargs = search_kwargs

        super().__init__(
            root,
            transform,
            pre_transform,
            pre_filter,
            force_reload=force_reload,
            download_only=download_only,
            **kwargs,
        )

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory."""
        return osp.join(self.root, "processed", f"{self.kwargs['k']}nn")

    @property
    def raw_file_names(self) -> list[str]:
        """Return the name of the downloaded files."""
        return ["CSG_tiny.csv"]

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
        import pandas as pd
        import torch
        from tqdm.auto import tqdm

        from src.graph import PeriodicKNN

        if self.download_only:
            return

        df = pd.read_csv(self.raw_paths[0])
        knn = PeriodicKNN(**self.kwargs)

        # Convert the target labels to consecutive 0-based indices
        unique_labels = sorted(set(df["SpaceGroupNumber"]))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        df["SpaceGroupNumber"] = df["SpaceGroupNumber"].map(label_to_index)

        data_list = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            data = knn.convert(row["Structure"])

            if data.num_nodes is None or data.num_nodes == 0:
                raise RuntimeError("The number of nodes in the graph is zero.")

            data.y = torch.full((data.num_nodes,), int(row["SpaceGroupNumber"]))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
