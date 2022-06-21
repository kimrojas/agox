import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.modules.databases import Database
from agox.modules.environments import Environment
from agox.modules.evaluators import LocalOptimizationEvaluator
from agox.modules.generators import RattleGenerator, RandomGenerator
from agox.modules.samplers import GeneticSampler, DistanceComparator
from agox.modules.collectors import StandardCollector
from agox.modules.models.descriptors.simple_fingerprint import SimpleFingerprint
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
database = Database(filename=db_path, order=3)

##############################################################################
# Search Settings:
##############################################################################

population_size = 10
descriptor = SimpleFingerprint(species=['Au', 'Ni'])
comparator = DistanceComparator(descriptor, threshold=0.5)
sampler = GeneticSampler(population_size=population_size, comparator=comparator, 
    order=4, database=database)
    
rattle_generator = RattleGenerator(**environment.get_confinement())
random_generator = RandomGenerator(**environment.get_confinement())
generators = [random_generator, rattle_generator]
num_candidates = {
    0:[population_size, 0], 
    5:[2, population_size-2], 
    10:[0, population_size]}

collector = StandardCollector(generators=generators, sampler=sampler, 
    environment=environment, num_candidates=num_candidates, order=1)

evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
    use_all_traj_info=False, optimizer_run_kwargs={'fmax':0.05, 'steps':400}, 
    order=2, constraints=environment.get_constraints(), 
    number_to_evaluate=population_size)

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(collector, database, evaluator, sampler)

agox.run(N_iterations=50)