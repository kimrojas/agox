import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.modules.databases import Database
from agox.modules.environments import Environment
from agox.modules.evaluators import LocalOptimizationEvaluator
from agox.modules.generators import RattleGenerator
from agox.modules.samplers import MetropolisSampler
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
db_path = 'db{}.db'.format(args.run_idx) # From input argument!
database = Database(filename=db_path, order=4)

##############################################################################
# Search Settings:
##############################################################################

sampler = MetropolisSampler(temperature=0.25, order=1, database=database)
    
rattle_generator = RattleGenerator(**environment.get_confinement(), 
    environment=environment, sampler=sampler, order=2)

evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
    use_all_traj_info=False, optimizer_run_kwargs={'fmax':0.05, 'steps':400}, 
    order=3, constraints=environment.get_constraints())

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(rattle_generator, database, evaluator, sampler)

agox.run(N_iterations=200)