import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
from ase.build import fcc100 

# Common AGOXs
from agox import AGOX
from agox.modules.databases import Database
from agox.modules.environments.environment_singular import EnvironmentSingular
from agox.modules.evaluators.local_optimization_evaluator import LocalOptimizationEvaluator
from agox.modules.collectors import TimeDependentCollector
from agox.modules.generators import StartGenerator, RattleGenerator
from agox.modules.postprocessors.constraints.box_constraint import BoxConstraint
from agox.modules.samplers.genetic_sampler import GeneticSampler, DistanceComparator

NUM_EPISODES = 50

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
rattle_generator = RattleGenerator(confinement_cell=confinement_cell, cell_corner=cell_corner)


generators = [start_generator, rattle_generator]

population_size = 10
num_samples = {0:[population_size, 0],
               2:[0, population_size]}

evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, optimizer_kwargs={'logfile':'-'}, optimizer_run_kwargs={'fmax':0.05, 'steps':400}, fix_template=True, constraints=[BC], number_to_evaluate=population_size)

# ################################################################################################
# # Ensemble / Sampler / Acquisitor
# ################################################################################################
from agox.modules.models.descriptors.simple_fingerprint import SimpleFingerprint

descriptor = SimpleFingerprint(species=['Pt', 'Au'])
comparator = DistanceComparator(descriptor, threshold=0.5)

sampler = GeneticSampler(database=database, comparator=comparator, population_size=population_size)

collector = TimeDependentCollector(generators=generators, sampler=sampler, environment=environment, num_samples=num_samples, report_timing=True)

################################################################################################
# Let get the show running! 
################################################################################################

agox = AGOX(environment=environment, db=database, collector=collector, sampler=sampler, 
            seed=run_idx, evaluator=evaluator)

agox.run(N_episodes=NUM_EPISODES)
