from pathlib import Path
import pytest 

import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator
from ase import Atoms

from agox.test.test_utils import TemporaryFolder, test_folder_path, compare_candidates

mode = 'lgpr_bh'
expected_path = Path(f'{test_folder_path}/run_tests/expected_outputs/{mode}_test/')

def test_agox_run(tmp_path, test_mode=True):

    with TemporaryFolder(tmp_path):
        
        # This loads the database file from the script file. 
        # This means that the documentation can link to this run-file.
        from agox.test.run_tests.script_lgpr_bh import database

        if test_mode: 
            test_candidates = database.get_all_candidates()
            test_energies = database.get_all_energies()
            test_forces = np.array([atoms.get_forces(apply_constraint=False) for atoms in database.get_all_candidates()])

            # Saved database:
            expected_database = Database(f'{expected_path}/db0.db')
            expected_database.restore_to_memory()
            expected_candidates = expected_database.get_all_candidates()
            expected_energies = expected_database.get_all_energies()
            expected_forces = np.array([atoms.get_forces(apply_constraint=False) for atoms in expected_database.get_all_candidates()])

            for candidate, expected_candidate in zip(test_candidates, expected_candidates):
                assert compare_candidates(candidate, expected_candidate), 'Candidates dont match.'

            assert len(expected_candidates) == len(test_candidates), 'Different numbers of candidates.'
            assert (test_energies == expected_energies).all(), 'Energies dont match.'
            assert np.allclose(test_forces, expected_forces), 'Forces dont match'


if __name__ == '__main__':

    import glob

    if not os.path.exists(expected_path):
        os.mkdir(expected_path)

    files = glob.glob(str(expected_path) + '/*')
    assert len(files) == 0, 'Expected path is not empty, delete the files if you are sure you want to remake them!'

    test_agox_run(expected_path, test_mode=False)