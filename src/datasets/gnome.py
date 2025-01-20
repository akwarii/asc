# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from torch_geometric.data import InMemoryDataset

from src.typing import PathLike


def download_from_link(link: str, output_dir: PathLike):
    """Download a file from a public link using requests."""
    import requests

    response = requests.get(link, timeout=10)
    if response.status_code == 200:
        with open(os.path.join(output_dir, os.path.basename(link)), "wb") as file:
            file.write(response.content)
            print(f"Downloaded {link} to {output_dir}")
    else:
        print(f"Failed to download {link}")


class Gnome(InMemoryDataset):
    """GNoME is a dataset of crystal structures predicted to be stable by the GNoME model
    created by the DeepMind team. The dataset contains ~380,000 crystal structures with space
    group numbers ranging from 1 to 230. The dataset is formatted as a CSV file with two
    columns: "MaterialId" and "Space Group Number". The "MaterialId" column contains the
    unique identifier of the material and the "Space Group Number" column contains the space
    group number of the crystal structure.

    Args:
        root: Root directory of the dataset.
        transform: A function that takes in a graph and returns a transformed version.
        pre_transform: A function that takes in a graph and returns a transformed version.
        pre_filter: A function that takes in a graph and returns a boolean value indicating
            whether the graph should be included in the dataset.
        force_reload: Whether to reload the dataset even if it already exists.
        kwargs: Additional keyword arguments to be passed to the KNNGraph or InMemoryDataset class.

    """

    API = "https://storage.googleapis.com/"
    BUCKET_NAME = "gdm_materials_discovery"
    FOLDER_NAME = "gnome_data"

    def __init__(
        self,
        root: str = "data/gnome",
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
        return ["stable_materials_summary.csv", "by_id.zip", "LICENSE"]

    @property
    def processed_file_names(self) -> list[str]:
        """Return the name of the processed files ie the transformed data saved to the disk."""
        return ["data.pt"]

    def download(self) -> None:
        """Download the dataset from Google and store it in the raw directory."""
        bucket_directory = os.path.join(self.API, self.BUCKET_NAME)
        parent_directory = os.path.join(bucket_directory, self.FOLDER_NAME)

        download_from_link(os.path.join(bucket_directory, "LICENSE"), self.raw_dir)

        for filename in self.raw_dir:
            public_link = os.path.join(parent_directory, filename)
            download_from_link(public_link, self.raw_dir)

        df = pd.read_csv(self.raw_paths[0], usecols=["MaterialId", "Space Group Number"])
        df.dropna().to_csv(self.raw_paths[0], index=False)

    def process(self) -> None:
        """Process the dataset by converting the structures to graphs, applying both pre-filter and
        pre-transform functions, and saving the processed data to disk. The data is saved in the
        processed directory as a single file named "data.pt".
        """
        import warnings
        from zipfile import ZipFile

        import torch
        from pymatgen.io.vasp.inputs import BadPoscarWarning
        from tqdm.auto import tqdm

        from src.graph import KNNGraph

        df = pd.read_csv(self.raw_paths[0])

        fnames = []
        unzipped_folder = Path(self.raw_dir) / "by_id"
        if not unzipped_folder.exists():
            with ZipFile(self.raw_paths[1], "r") as zipfile:
                zipfile.extractall(self.raw_dir)

        # Filter out the files that are not in the stable materials summary. This is to ensure that
        # the data and targets are aligned (not the case for the original dataset). Some files are
        # missing in the stable materials summary, so we need to check if the file exists.
        fnames = [
            unzipped_folder / f"{fid}.CIF"
            for fid in df["MaterialId"].to_list()
            if os.path.isfile(unzipped_folder / f"{fid}.CIF")
        ]

        raw_data_list = [f.read_text() for f in fnames]
        target_list = df["Space Group Number"].values.tolist()

        # Convert the target labels to consecutive 0-based indices
        unique_targets = sorted(set(target_list))
        label_to_index = {label: idx for idx, label in enumerate(unique_targets)}
        target_list = [label_to_index[target] for target in target_list]

        knn = KNNGraph(**self.kwargs)

        data_list = []
        for raw_data, target in tqdm(zip(raw_data_list, target_list), total=len(raw_data_list)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BadPoscarWarning)
                data = knn.convert(raw_data)

            if data.num_nodes is None or data.num_nodes == 0:
                raise RuntimeError("The number of nodes in the graph is zero.")

            data.y = torch.full((data.num_nodes,), target - 1, dtype=torch.long)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
