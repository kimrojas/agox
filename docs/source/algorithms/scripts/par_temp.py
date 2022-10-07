import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.databases import ConcurrentDatabase
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RattleGenerator
from agox.samplers import ParallelTemperingSampler
from ase import Atoms

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-i', '--run_idx', type=int, default=0)
args = parser.parse_args()

##############################################################################
# Calculator
##############################################################################

from ase.calculators.emt import EMT

calc = EMT()

##############################################################################    
# System & general settings:
##############################################################################
    
template = Atoms('', cell=np.eye(3)*12)
confinement_cell = np.eye(3) * 8
confinement_corner = np.array([3, 3, 3])
environment = Environment(template=template, symbols='Au8Ni8', 
    confinement_cell=confinement_cell, confinement_corner=confinement_corner)

# Database
total_workers = 4
sync_frequency = 10
run_idx = args.run_idx
database_index = (run_idx-1) // total_workers + 1
worker_index = (run_idx-1) % total_workers
main_directory = '/home/machri/Projects/agox/data/documentation/'
sub_directory = os.path.splitext(os.path.basename(__file__))[0] + '/'
db_path = main_directory + sub_directory + 'con_db{}.db'.format(database_index)
database = ConcurrentDatabase(filename=db_path, store_meta_information=True, 
    write_frequency=1, worker_number=worker_index, total_workers=total_workers, 
    sync_frequency=sync_frequency, order=4, sleep_timing=0.1)

# ##############################################################################
# # Search Settings:
# ##############################################################################

temperatures = [0.1*1.5**power for power in range(total_workers)]
sampler = ParallelTemperingSampler(temperatures=temperatures, order=3, 
    database=database, swap_order=5)
    
rattle_generator = RattleGenerator(**environment.get_confinement(), 
    environment=environment, sampler=sampler, order=1)

evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
    use_all_traj_info=False, optimizer_run_kwargs={'fmax':0.05, 'steps':400}, 
    order=2, constraints=environment.get_constraints())

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(rattle_generator, database, evaluator)

agox.run(N_iterations=50)