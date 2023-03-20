from agox.test.test_utils import TemporaryFolder, test_folder_path, check_folder_is_empty

def test_tracker(tmp_path, cmd_options):

        with TemporaryFolder(tmp_path):
            from agox.tracker import Tracker
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
            from argparse import ArgumentParser

            # Manually set seed and database-index
            seed = 42
            database_index = 0

            # Using argparse if e.g. using array-jobs on Slurm to do several independent searches. 
            # parser = ArgumentParser()
            # parser.add_argument('-i', '--run_idx', type=int, default=0)
            # args = parser.parse_args()

            # seed = args.run_idx
            # database_index = args.run_idx

            ##############################################################################
            # Calculator
            ##############################################################################

            from ase.calculators.emt import EMT

            calc = EMT()

            ##############################################################################    
            # System & general settings:
            ##############################################################################
                
            template = Atoms('', cell=np.eye(3)*12)
            confinement_cell = np.eye(3) * 6
            confinement_corner = np.array([3, 3, 3])
            environment = Environment(template=template, symbols='Au8Ni8', 
                confinement_cell=confinement_cell, confinement_corner=confinement_corner)

            # Database
            db_path = 'db{}.db'.format(database_index) # From input argument!
            database = Database(filename=db_path, order=3, write_frequency=1)

            ##############################################################################
            # Search Settings:
            ##############################################################################
                
            random_generator = RandomGenerator(**environment.get_confinement(), 
                environment=environment, order=1)

            random_generator.add_tracked_attribute('set_key')
            key = f'{random_generator.__name__}.set_key'

            # Wont relax fully with steps:5 - more realistic setting would be 100+.
            evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
                optimizer_run_kwargs={'fmax':0.05, 'steps':5}, store_trajectory=False,
                order=2, constraints=environment.get_constraints())

            ##############################################################################
            # Let get the show running! 
            ##############################################################################

            agox = AGOX(random_generator, database, evaluator, seed=seed)

            # State tracking: 
            def mean_position(list_of_atoms):
                positions = np.array([atoms.get_positions() for atoms in list_of_atoms])
                return positions.mean()

            # Track the mean position of the atoms in the system.
            agox.tracker.add_state_tracker(mean_position, 'mean_position', 'evaluated_candidates')
            agox.tracker.add_state_tracker(mean_position, 'mean_position', 'candidates')

            agox.run(N_iterations=10)

            loaded_properties = Tracker.load('db0_tracker.npz')

        tracker = agox.tracker

        assert key in tracker.tracked_attributes.keys()

        for key in tracker.tracked_attributes.keys():
            assert key in loaded_properties.keys()
            assert (tracker.tracked_attributes[key] == loaded_properties[key]).all()
            assert len(tracker.tracked_attributes[key]) == 10 # 10 iterations
