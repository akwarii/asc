import numpy as np
from pymatgen.core import Structure
from src.utils.neighbors import find_knn_in_shell

NACL_POSCAR = """
1.0
3.4220150000000000    0.0000000000000000    1.9757020000000001
1.1406710000000000    3.2263060000000001    1.9757020000000001
0.0000000000000000    0.0000000000000000    3.9514019999999999
Na Cl
1 1
direct
0.0000000000000000    0.0000000000000000    0.0000000000000000 Na
0.5000000000000000    0.5000000000000000    0.5000000000000000 Cl
"""
FIRST_SHELL_DIST = 2.794063078217634

def test_find_knn_in_shell():
    structure = Structure.from_str(NACL_POSCAR, fmt="poscar")
    
    # Test case 1: Basic test with a simple structure
    rcut = 3.0
    n_neighbors = 4
    expected_output = [
        [FIRST_SHELL_DIST,] * 4,
        [FIRST_SHELL_DIST,] * 4,
    ]
    output = find_knn_in_shell(structure, rcut, n_neighbors)
    nn_distance = [[s.nn_distance for s in sites] for sites in output]
    assert np.allclose(nn_distance, expected_output)

    # Test case 2: Test with a larger structure and different cutoff radius
    rcut = 4.5
    n_neighbors = 3
    expected_output = [
        [FIRST_SHELL_DIST,] * 3,
        [FIRST_SHELL_DIST,] * 3,
    ]
    output = find_knn_in_shell(structure, rcut, n_neighbors)
    nn_distance = [[s.nn_distance for s in sites] for sites in output]
    assert np.allclose(nn_distance, expected_output)

    # Test case 3: Test with a structure that has missing neighbors
    rcut = 2.0
    n_neighbors = 4
    expected_output = [
        [FIRST_SHELL_DIST,] * 4,
        [FIRST_SHELL_DIST,] * 4,
    ]
    output = find_knn_in_shell(structure, rcut, n_neighbors)
    nn_distance = [[s.nn_distance for s in sites] for sites in output]
    assert np.allclose(nn_distance, expected_output)
