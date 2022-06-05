import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
from ase.build import fcc100 

# Common AGOXs
from agox import AGOX
from agox.modules.databases import Database
from agox.modules.environments.environment_singular import EnvironmentSingular
from agox.modules.evaluators.local_optimization import LocalOptimizationEvaluator
from agox.modules.collectors import TimeDependentCollector
from agox.modules.generators import StartGenerator
from agox.modules.postprocessors.constraints.box_constraint import BoxConstraint

NUM_iterationS = 500

################################################################################################
# Input arguments
################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', type=int, default=0)
args = parser.parse_args()    
run_idx = args.index

################################################################################################
# Calculator
################################################################################################

from ase.calculators.emt import EMT
calc = EMT()

################################################################################################
# General settings:
################################################################################################

template = fcc100('Au', (6, 6, 3), vacuum=10)
environment = EnvironmentSingular(template=template, symbols='Pt14')

# Database
db_path = 'db{}.db'.format(run_idx)
database = Database(filename=db_path)

################################################################################################
# Generator settings:
################################################################################################

fractions = np.array([0.7, 0.7, 0])
confinement_cell = template.get_cell() * fractions.T
cell_corner = template.get_cell() @ np.array([0.15, 0.15, 0])
cell_corner[2] = np.max(template.positions[:, 2])-1
confinement_cell[2, 2] = 10

BC = BoxConstraint(confinement_cell, cell_corner, indices=environment.get_missing_indices())
confinement_limits = BC.get_confinement_limits()

start_generator = StartGenerator(confinement_cell=confinement_cell, cell_corner=cell_corner, may_nucleate_at_several_places=True)

generators = [start_generator]
num_samples = {0:[1]}

gauges = [LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, optimizer_kwargs={'logfile':'-'}, optimizer_run_kwargs={'fmax':0.05, 'steps':400}, fix_template=True, constraints=[BC])]

# ################################################################################################
# # Ensemble / Sampler / Acquisitor
# ################################################################################################

collector = TimeDependentCollector(generators=generators, sampler=None, environment=environment, num_samples=num_samples, report_timing=True)

################################################################################################
# Let get the show running! 
################################################################################################

agox = AGOX(environment=environment, db=database, collector=collector, 
            seed=run_idx, gauges=gauges[0])

agox.run(N_iterations=NUM_iterationS)
