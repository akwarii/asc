import hashlib
import re
from collections.abc import Sequence
from itertools import zip_longest
from pathlib import Path

import numpy as np

from src.typing import PathLike


def md5(fname):
    hash_md5 = hashlib.md5(usedforsecurity=False)
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def check_md5(fname, md5_checksum):
    return md5(fname) == md5_checksum


def check_integrity(fpaths: Sequence[PathLike], checksums: Sequence[str | None]) -> bool:
    """Check the integrity of the dataset."""
    for fpath, md5 in zip_longest(fpaths, checksums):
        if isinstance(fpath, str):
            fpath = Path(fpath)

        if not fpath.is_file():
            return False
        if md5 is None:
            continue
        if not check_md5(fpath, md5):
            return False
    return True


def lattice_from_geometry(geometry):
    """Create a cell matrix from its lattice parameter (a,b,c,alpha,beta,gamma). The returned cell
    is orientated such that a and b are normal to (0,0,1) and a is parallel to (1,0,0). The cell
    vectors are defined row-wise. This implementation is based on the one in ASE.

    Args:
        geometry (Sequence): A sequence of lattice parameters (a,b,c,alpha,beta,gamma).

    Returns:
        np.ndarray (3x3): The cell matrix.
    """
    a, b, c, alpha, beta, gamma = geometry
    alpha, beta, gamma = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)

    # Define rotated X,Y,Z-system, with Z along (0,0,1) and X along
    # the projection of a_direction onto the normal plane of Z.
    z_normed = np.array((0, 0, 1))

    ad = np.array((1, 0, 0))
    x = ad - np.dot(ad, z_normed) * z_normed
    x_normed = x / np.linalg.norm(x)

    y_normed = np.cross(z_normed, x_normed)

    # Define the cosines and sines
    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    # Build the cell vectors
    va = a * np.array([1, 0, 0])
    vb = b * np.array([cos_gamma, sin_gamma, 0])
    cx = cos_beta
    cy = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz_sqr = 1.0 - cx * cx - cy * cy
    assert cz_sqr >= 0
    cz = np.sqrt(cz_sqr)
    vc = c * np.array([cx, cy, cz])

    # Convert to the Cartesian x,y,z-system
    abc = np.vstack((va, vb, vc))
    T = np.vstack((x_normed, y_normed, z_normed))
    cell = np.dot(abc, T)

    return cell


def poscar_from_entry(entry):
    assert "compound" in entry, "Entry must contain a `compound` key"
    assert "geometry" in entry, "Entry must contain a `geometry` key"
    assert "positions_fractional" in entry, "Entry must contain a `positions_fractional` key"

    lattice = lattice_from_geometry(entry["geometry"])
    species = re.findall("[A-Z][a-z]*", entry["compound"])
    n_atoms_per_species = re.findall(r"\d+", entry["compound"])

    poscar = f"{entry['compound']}\n"
    poscar += "1.0\n"
    poscar += f"{lattice[0][0]:.10f} {lattice[0][1]:.10f} {lattice[0][2]:.10f}\n"
    poscar += f"{lattice[1][0]:.10f} {lattice[1][1]:.10f} {lattice[1][2]:.10f}\n"
    poscar += f"{lattice[2][0]:.10f} {lattice[2][1]:.10f} {lattice[2][2]:.10f}\n"
    poscar += " ".join(species) + "\n"
    poscar += " ".join(n_atoms_per_species) + "\n"
    poscar += "Direct\n"
    for position in entry["positions_fractional"]:
        poscar += f"{position[0]:.10f} {position[1]:.10f} {position[2]:.10f}\n"

    return poscar
