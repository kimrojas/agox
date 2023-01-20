import os
import numpy as np

class TemporaryFolder:

    def __init__(self, path):
        d = path / "temp_path"
        d.mkdir()
        self.start_dir = os.getcwd()
        self.path = d

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, *args):
        os.chdir(self.start_dir)

def compare_candidates(atoms_1, atoms_2):
    pos_bool = np.allclose(atoms_1.positions, atoms_2.positions)
    cell_bool = np.allclose(atoms_1.cell, atoms_2.cell)
    numbers_bool = np.allclose(atoms_1.numbers, atoms_2.numbers)
    return pos_bool * cell_bool * numbers_bool
