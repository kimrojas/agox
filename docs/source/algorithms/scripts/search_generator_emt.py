import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np

# Common AGOXs
from agox.main import AGOX
from agox.modules.environments import Environment
from agox.modules.evaluators import SinglePointEvaluator
from agox.modules.databases import Database

# AGOX GENERATOR
from agox.modules.generators.agox import AGOXGenerator
from agox.modules.collectors import StandardCollector

# Local GPR model
from agox.modules.models.descriptors.soap import SOAP
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from agox.modules.models.local_GPR.LSGPR_CUR import LSGPRModelCUR
from agox.modules.models.priors.repulsive import Repulsive
from ase.io import read

NUM_ITERATIONS = 50
C1, C2 = 0.7, 3

################################################################################
# Input arguments
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', type=int, default=0)
args = parser.parse_args()    
run_idx = args.index

################################################################################
# Calculator
################################################################################

from ase.calculators.emt import EMT

calc = EMT()

################################################################################
# Environment Settings:
################################################################################

template = Atoms('', cell=np.eye(3)*12)
confinement_cell = np.eye(3) * 8
confinement_corner = np.array([3, 3, 3])
environment = Environment(template=template, symbols='Au8Ni8', 
    confinement_cell=confinement_cell, confinement_corner=confinement_corner)

constraints = environment.get_constraints()

################################################################################
# Database Settings:
################################################################################

db_path = 'db{}.db'.format(run_idx)
database = Database(filename=db_path, order=6, write_frequency=1)

################################################################################
# Model settings:
################################################################################

transfer_data = <TRANSFER_DATA>
descriptor = SOAP(environment.get_all_species(), r_cut=5., nmax=3, lmax=2, 
    sigma=1, weight=True, periodic=True)
kernel = C(1)*RBF(length_scale=20)
local_model = LSGPRModelCUR(database=database, kernel=kernel, 
    descriptor=descriptor, noise=0.01, prior=Repulsive(ratio=C1), verbose=True,
    iteration_start_training=0, transfer_data=transfer_data,
    use_prior_in_training=True)
local_model.train_model(training_data=[])

################################################################################
# AGOX Generator:
################################################################################

# Random, permute, rattle, sampling
number_of_candidates = [4, 8, 12, 0]

agox_generator = AGOXGenerator.get_gofee_generator(environment, database, 
    local_model, iterations=150, c1=C1, c2=C2, 
    number_of_candidates=number_of_candidates,
    constraints=constraints, fix_template=False, 
    model_kwargs={'use_delta_in_training':True})

################################################################################
# 'Real' stuff:
################################################################################
from agox.modules.acquisitors.kmeans import KmeansAcquisitor
from agox.modules.models.descriptors.simple_fingerprint import SimpleFingerprint
from agox.modules.postprocessors.relax import RelaxPostprocess

K = 10
descriptor = SimpleFingerprint(species=environment.get_all_species())
acquisitor = KmeansAcquisitor(descriptor, local_model, k=K, order=3)

collector = StandardCollector(generators=[agox_generator], 
    sampler=None, environment=environment, num_candidates={0:[1]})

relaxer = RelaxPostprocess(model=local_model, 
        optimizer_run_kwargs={'steps':15, 'fmax':0.1},
        constraints=constraints, gets={'get_key':'prioritized_candidates'}, 
        sets={'set_key':'opt_prioritized_candidates'}, order=4)

evaluator = SinglePointEvaluator(calc, number_to_evaluate=K, 
    gets={'get_key':'opt_prioritized_candidates'}, order=5)

################################################################################
# Let get the show running! 
################################################################################
agox = AGOX(database, collector, relaxer, acquisitor, evaluator, seed=run_idx)

agox.run(N_iterations=NUM_ITERATIONS)

