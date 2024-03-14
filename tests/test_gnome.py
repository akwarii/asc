from pathlib import Path
from unittest.mock import patch

from src.data.datasets import Gnome


@patch("src.data.datasets.gnome.Gnome.check_exists", return_value=True)
@patch("src.data.datasets.gnome.Gnome.load", return_value=(["data"], ["targets"]))
@patch("src.data.datasets.gnome.Gnome.download")
def test_gnome_init(mock_exists, mock_data, mock_download):
    gnome = Gnome(root="data", download=False)

    assert gnome.root == Path("data")
    gnome.download.assert_not_called()

    gnome.__init__(root="data", download=True)  # Call __init__ explicitly
    gnome.download.assert_called_once()


@patch("src.data.datasets.gnome.Gnome.check_exists", return_value=True)
@patch("src.data.datasets.gnome.Gnome.load", return_value=(["1", "2", "3"], [1, 2, 3]))
def test_gnome_len(mock_exists, mock_data):
    gnome = Gnome(root="data", download=False)
    assert len(gnome) == 3
