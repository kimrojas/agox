import os
import numpy as np
from importlib_resources import files

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

    if atoms_1 is None and atoms_2 is None:
        return True

    pos_bool = np.allclose(atoms_1.positions, atoms_2.positions)
    cell_bool = np.allclose(atoms_1.cell, atoms_2.cell)
    numbers_bool = np.allclose(atoms_1.numbers, atoms_2.numbers)
    return pos_bool * cell_bool * numbers_bool

test_folder_path = os.path.join(files('agox'), 'test/')

test_data_dicts = [
    {'path':'datasets/AgO-dataset.traj', 'remove':6, 'name':'AgO'}, 
    {'path':'datasets/B12-dataset.traj', 'remove':12, 'name':'B12'},        
    {'path':'datasets/C30-dataset.traj', 'remove':30, 'name':'C30'},
    ]

for dictionary in test_data_dicts:
    dictionary['path'] = os.path.join(test_folder_path, dictionary['path'])
