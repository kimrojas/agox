import matplotlib
matplotlib.use('Agg')

import numpy as np
from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator, RattleGenerator
from agox.samplers import KMeansSampler
from agox.collectors import ParallelCollector
from agox.models import ModelGPR
from agox.acquisitors import LowerConfidenceBoundAcquisitor
from agox.postprocessors import ParallelRelaxPostprocess
from ase import Atoms
from agox.models.descriptors.exponential_density import ExponentialDensity
from agox.generators.ce_generator import ComplementaryEnergyGenerator
from agox.generators.complementary_energy.ce_calculators import ComplementaryEnergyDistanceCalculator
from agox.generators.complementary_energy.attractor_methods.ce_attractors_current_structure import AttractorCurrentStructure

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
db_path = 'db{}.db'.format(0) # From input argument!
database = Database(filename=db_path, order=6, write_frequency=1)

##############################################################################
# Search Settings:
##############################################################################

model = ModelGPR.default(environment, database)

sample_size = 10
descriptor = model.get_descriptor()
descriptor.use_cache = True
sampler = KMeansSampler(descriptor=descriptor, database=database, 
            sample_size=sample_size)

rattle_generator = RattleGenerator(**environment.get_confinement())
random_generator = RandomGenerator(**environment.get_confinement())

lambs = [0.5, 1, 1.5]
rc = 10.
desc = ExponentialDensity(['Au', 'Ni'], lambs = lambs, rc = rc)
ce_calc = ComplementaryEnergyDistanceCalculator(descriptor = desc)
ce_attractors = AttractorCurrentStructure(desc, attractors_from_template = False)
ce_attractors.attach(database)
ce_generator = ComplementaryEnergyGenerator(ce_calc, desc, ce_attractors, **environment.get_confinement())

# Dict specificies how many candidates are created with and the dict-keys are iterations. 
generators = [random_generator, rattle_generator, ce_generator]
num_candidates = {0:[10, 0], 5:[3, 6, 1]}

acquisitor = LowerConfidenceBoundAcquisitor(model_calculator=model, 
    kappa=2, order=4)

# CPU-count is set here for Ray - leave it out to use as many cores as are available. 
collector = ParallelCollector(generators=generators, sampler=sampler,
    environment=environment, num_candidates=num_candidates, order=2, 
    cpu_count=5)
    
# Number of steps is very low - should be set higher for a real search!
relaxer = ParallelRelaxPostprocess(model=acquisitor.get_acquisition_calculator(), 
    constraints=environment.get_constraints(), order=3, start_relax=8, 
    optimizer_run_kwargs={'steps':5})

evaluator = LocalOptimizationEvaluator(calc, 
    gets={'get_key':'prioritized_candidates'}, 
    optimizer_kwargs={'logfile':None}, store_trajectory=True,
    optimizer_run_kwargs={'fmax':0.05, 'steps':1}, order=5)

##############################################################################
# Let get the show running! 
##############################################################################
    
agox = AGOX(collector, acquisitor, relaxer, database, evaluator, seed=seed)

agox.run(N_iterations=10)
