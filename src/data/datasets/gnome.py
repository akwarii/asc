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
from zipfile import ZipFile

import pandas as pd
import requests
from pymatgen.core import Structure

from src.data.datasets.base import GraphDataset
from src.utils.constants import GNOME_CLASSES
from src.utils.typing import PathLike


def download_from_link(link: str, output_dir: PathLike):
    """Download a file from a public link using requests."""
    response = requests.get(link, timeout=10)
    if response.status_code == 200:
        with open(os.path.join(output_dir, os.path.basename(link)), "wb") as file:
            file.write(response.content)
            print(f"Downloaded {link} to {output_dir}")
    else:
        print(f"Failed to download {link}")


class Gnome(GraphDataset):
    """GNoME is a dataset of crystal structures predicted to be stable by the GNoME model
    created by the DeepMind team. The dataset contains ~380,000 crystal structures with space
    group numbers ranging from 1 to 230. The dataset is formatted as a CSV file with two
    columns: "MaterialId" and "Space Group Number". The "MaterialId" column contains the
    unique identifier of the material and the "Space Group Number" column contains the space
    group number of the crystal structure.

    Args:
        root (str): Root directory of the dataset.
        transform (Callable | None): A function/transform that takes in a graph and returns a transformed version.
        struct_transform (Callable | None): A function/transform that takes in a structure and returns a transformed version.
        target_transform (Callable | None): A function/transform that takes in a target and returns a transformed version.
        download (bool): Whether to download the dataset if it doesn't exist.
        load (bool): Whether to load the dataset.
        chunk_size (int): Number of entries of each chunk to download.
        **graph_kwargs: Additional keyword arguments to be passed to the Graph class.

    Attributes:
        API (AflowAPI): URL to the google storage api.
        classes (list): A list of space group numbers ranging from 1 to 230.
        resources (list): Names of the files containing the dataset.

    Methods:
        __getitem__: Retrieves a graph and its corresponding target from the dataset.
        __len__: Returns the length of the dataset.
        load: Loads the data from the resource files.
        download: Downloads the Aflow dataset if it doesn't exist already.
    """

    API = "https://storage.googleapis.com/"
    _BUCKET_NAME = "gdm_materials_discovery"
    _FOLDER_NAME = "gnome_data"

    classes = GNOME_CLASSES

    # Note that other datasets exists in the same bucket
    resources = (
        "stable_materials_summary.csv",
        "by_id.zip",
    )

    def __init__(
        self,
        root: str,
        transform: Callable | None = None,
        struct_transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
        graph_kwargs: dict[str, Any] = {},
    ) -> None:
        super().__init__(root, transform, struct_transform, target_transform, graph_kwargs)

        if download:
            self.download()

        if not self.check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self.load()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        fname, target = self.data[index], self.targets[index]

        struct = Structure.from_file(fname, fmt="cif")
        if self.struct_transform is not None:
            struct = self.struct_transform(struct)

        graph = self.knn.convert(struct)
        if self.transform is not None:
            graph = self.transform(graph)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return graph, target

    def __len__(self) -> int:
        return len(self.data)

    def load(self) -> tuple[list[Path], list[int]]:
        df = pd.read_csv(self.raw_folder / "stable_materials_summary.csv")
        targets = df["Space Group Number"].values.tolist()

        # Load the zip file and extract the contents
        data = []
        unzipped_folder = self.raw_folder / "by_id"
        if not unzipped_folder.exists():
            with ZipFile(self.raw_folder / "by_id.zip", "r") as zip:
                zip.extractall(self.raw_folder)

        # Filter out the files that are not in the stable materials summary.
        # This is to ensure that the data and targets are aligned (not the case for the original dataset)
        data = [unzipped_folder / f"{fid}.CIF" for fid in df["MaterialId"].to_list()]

        return data, targets

    def download(self) -> None:
        if self.check_exists():
            print(f"Dataset already exists at {self.root}")
            return

        self.raw_folder.mkdir(parents=True, exist_ok=True)

        bucket_directory = os.path.join(self.API, self._BUCKET_NAME)
        parent_directory = os.path.join(bucket_directory, self._FOLDER_NAME)

        # Download LICENSE file
        download_from_link(os.path.join(bucket_directory, "LICENSE"), self.raw_folder)

        # Download data files.
        print(f"Downloading Gnome data from {parent_directory} to {self.raw_folder}...")
        for filename in self.resources:
            public_link = os.path.join(parent_directory, filename)
            download_from_link(public_link, self.raw_folder)

        df: pd.DataFrame = pd.read_csv(
            self.raw_folder / "stable_materials_summary.csv",
            usecols=["MaterialId", "Space Group Number"],
        )
        df.dropna().to_csv(self.raw_folder / "stable_materials_summary.csv", index=False)
