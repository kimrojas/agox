import matplotlib

matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.modules.collectors import TimeDependentCollector
from agox.modules.databases import Database
from agox.modules.environments.environment_singular import EnvironmentSingular
from agox.modules.evaluators.local_optimization_evaluator import LocalOptimizationEvaluator
from agox.modules.generators import StartGenerator
from agox.modules.postprocessors.constraints.box_constraint import BoxConstraint
from ase import Atoms

NUM_EPISODES = 50

################################################################################################
# Calculator
################################################################################################

from ase.calculators.emt import EMT

calc = EMT()

################################################################################################
# General settings:
################################################################################################

template = Atoms('', cell=np.eye(3)*12)
environment = EnvironmentSingular(template=template, symbols='Au5Ni')

# Database
db_path = 'db1.db'
database = Database(filename=db_path)

################################################################################################
# Search Settings:
################################################################################################

confinement_cell = np.eye(3) * 6
cell_corner = np.array([3, 3, 3])
start_generator = StartGenerator(confinement_cell=confinement_cell, cell_corner=cell_corner, may_nucleate_at_several_places=True)

generators = [start_generator]
num_samples = {0:[1]}

gauges = [LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, optimizer_kwargs={'logfile':None}, optimizer_run_kwargs={'fmax':0.05, 'steps':400})]

collector = TimeDependentCollector(generators=generators, sampler=None, environment=environment, num_samples=num_samples, report_timing=True)

################################################################################################
# Let get the show running! 
################################################################################################

agox = AGOX(environment=environment, db=database, collector=collector, gauges=gauges)

agox.run(N_episodes=NUM_EPISODES)
