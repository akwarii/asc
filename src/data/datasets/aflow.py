import codecs
import json
from pathlib import Path
import sys
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

from torchvision.datasets.utils import _flip_byte_order, check_integrity
from torch.utils.data import Dataset
from src.api.aflow import AflowAPI


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


SN3_PASCALVINCENT_TYPEMAP = {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1) : 4 * (i + 2)]) for i in range(nd)]

    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))

    # The MNIST format uses the big endian byte order, while `torch.frombuffer` uses whatever the system uses. In case
    # that is little endian and the dtype has more than one byte, we need to flip them.
    if sys.byteorder == "little" and parsed.element_size() > 1:
        parsed = _flip_byte_order(parsed)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)


def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 1:
        raise ValueError(f"x should have 1 dimension instead of {x.ndimension()}")
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 3:
        raise ValueError(f"x should have 3 dimension instead of {x.ndimension()}")
    return x


class Aflow(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    api = AflowAPI()

    classes = list(range(1, 231)) # space groups numbers

    resources = [f"data_{class_idx}.json" for class_idx in classes]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self.root = root

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        #TODO: I don't think the target must be casted to int
        graph, target = self.data[index], int(self.targets[index])
        
        #TODO: Create a Graph instance

        if self.transform is not None:
            graph = self.transform(graph)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return graph, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return Path(self.root, self.__class__.__name__, "raw")

    @property
    def processed_folder(self) -> str:
        return Path(self.root, self.__class__.__name__, "processed")
    
    #TODO: implement load data
    #! This method must load the files, not the graph representations of the data
    def _load_data(self):
        pass

    def _check_exists(self) -> bool:
        return all(
            Path(self.raw_folder, fname).is_file()
            for fname in self.resources
        )

    def download(self, chunk_size: int = 100_000) -> None:
        """Download the Aflow dataset if it doesn't exist already."""

        if self._check_exists():
            return

        Path(self.raw_folder).mkdir(parents=True, exist_ok=True)
        
        for class_idx in range(1, 231):
            file = Path(self.raw_folder, f"data_{class_idx}.json")
            
            if file.is_file() and file.stat().st_size > 0:
                continue
            
            print(f"Downloading Aflow data for space group {class_idx}")
            
            # Download data by chunks to avoid server timeout
            page_number = 1
            current_data = None
            total_data = []
            
            with self.api as aflow_api:
                while current_data != []:
                    current_data = aflow_api.request(f"spacegroup_relax({class_idx})", paging_range=(page_number, chunk_size))

                    page_number += 1
                    total_data += current_data
                
                for entry in total_data:
                    del entry["Pearson_symbol_relax"]
                    del entry["compound"]
                    
                    # If CONTCAR.relax is not available, remove entry
                    try:
                        entry["CONTCAR.relax"] = aflow_api.get_contcar(entry)
                    except RuntimeError:
                        del entry
                
            file.write_text(json.dumps(total_data, sort_keys=True, indent=4))
            
    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
    
    
if __name__ == "__main__":
    Aflow(root="data", download=True)