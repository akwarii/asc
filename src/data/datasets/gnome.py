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
from typing import Any
from zipfile import ZipFile

import pandas as pd
import requests
from pymatgen.core import Structure

from src.data.datasets.base_dataset import CrystalGraphDataset
from src.processing.graph import Graph


def download_from_link(link: str, output_dir: str):
    """Download a file from a public link using requests."""
    response = requests.get(link, timeout=10)
    if response.status_code == 200:
        with open(os.path.join(output_dir, os.path.basename(link)), "wb") as file:
            file.write(response.content)
            print(f"Downloaded {link} to {output_dir}")
    else:
        print(f"Failed to download {link}")


class Gnome(CrystalGraphDataset):
    API = "https://storage.googleapis.com/"
    _BUCKET_NAME = "gdm_materials_discovery"
    _FOLDER_NAME = "gnome_data"

    classes = list(range(1, 231))  # space groups numbers

    # Note that other datasets exists in the same bucket
    resources = (
        "stable_materials_r2scan.csv",
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
        load: bool = True,
        **graph_kwargs,
    ) -> None:
        super().__init__(root, transform, struct_transform, target_transform)
        self.graph_kwargs = graph_kwargs

        if download:
            self.download()

        if not self.check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if load:
            self.data, self.targets = self._load_data()

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        if not self.data:
            RuntimeWarning("Dataset not loaded. Use load=True to load the dataset")
            return None, None
        
        fname, target = self.data[index], self.targets[index]

        with ZipFile(self.raw_folder / "by_id.zip", "r") as zip_ref:
            with zip_ref.open(fname) as file:
                cif = file.read().decode("utf-8")

        struct = Structure.from_str(cif, fmt="cif")

        if self.struct_transform is not None:
            struct = self.struct_transform(struct)

        # TODO: really need to refactor Graph to a graph factory to improve efficiency
        # and if possible use DGL/PyG graphs instead of custom implementation
        graph = Graph(**self.graph_kwargs)
        graph.set_features(struct)

        if self.transform is not None:
            graph = self.transform(graph)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return graph, target

    def __len__(self) -> int:
        return len(self.data)

    # TODO: efficient implementation of _load_data
    # as of now, there is a mismatch between the data and targets
    def _load_data(self) -> tuple[list[str], list[int]]:
        df = pd.read_csv(self.raw_folder / "stable_materials_summary.csv")
        targets = df["Space Group Number"].values.tolist()

        # Load the zip file and extract the contents
        data = []
        with ZipFile(self.raw_folder / "by_id.zip", "r") as zip_ref:
            data = [f for f in zip_ref.namelist() if f.endswith(".CIF")]

        # assert len(data) == len(targets), "Data and targets length mismatch"

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
        for filename in self.resources:
            public_link = os.path.join(parent_directory, filename)
            download_from_link(public_link, self.raw_folder)

        print(f"Done downloading data to directory: {self.root}")


if __name__ == "__main__":
    gnome = Gnome(root="data", download=True)
    print(gnome[0])
