import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
from ase.build import fcc100 

# Common AGOXs
from agox import AGOX
from agox.modules.databases import Database
from agox.modules.environments.environment_singular import EnvironmentSingular
from agox.modules.acquisitors.acquisitor_LCB import LowerConfidenceBoundAcquisitor
from agox.modules.models.model_GPR import ModelGPR
from agox.modules.models.gaussian_process.default_model_gofee import get_default_GPR_model
from agox.modules.evaluators.energy_evaluator import EnergyEvaluator
from agox.modules.collectors import TimeDependentCollector
from agox.modules.generators import RattleGenerator, StartGenerator, CenterOfGeometryGenerator
from agox.modules.postprocessors.postprocess_io_relax import PrimaryPostProcessIORelax
from agox.modules.samplers.sampler_kmeans import SamplerKMeans
from agox.modules.postprocessors.constraints.box_constraint import BoxConstraint

KAPPA = 2
NUM_EPISODES = 100

################################################################################################
# Input arguments
################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', type=int, default=0)
parser.add_argument('-t', '--task_per_node', type=int, default=1)
args = parser.parse_args()    
run_idx = args.index

################################################################################################
# Calculator
################################################################################################

from ase.calculators.emt import EMT

calc = EMT()
evaluator = EnergyEvaluator(calc)

################################################################################################
# General settings:
################################################################################################

template = fcc100('Au', (6, 6, 3), vacuum=10)
environment = EnvironmentSingular(template=template, symbols='Pt14')

# Database
db_path = 'db{}.db'.format(run_idx)
database = Database(filename=db_path)


lambda1min = 1; lambda1max = 20; lambda1ini = (lambda1max - lambda1min)/2 + lambda1min
lambda2min = 1e-1; lambda2max = 1; lambda2ini = (lambda2max - lambda2min)/2 + lambda2min
theta0min = 1e0; theta0max = 1e5; theta0ini = 5000                         

hyperparams = {'lambda1min':lambda1min, 'lambda1max':lambda1max, 'lambda1ini':lambda1ini,
               'lambda2min':lambda2min, 'lambda2max':lambda2max, 'lambda2ini':lambda2ini, 
               'theta0min':theta0min, 'theta0max':theta0max, 'theta0ini':theta0ini}
            
model, feature_calc = get_default_GPR_model(environment=environment, use_delta_func=True, return_feature_calc=True, **hyperparams)
model_calculator = ModelGPR(model, update_interval=1, optimize_frequency=25, optimize_loglikelyhood=True, use_saved_features=True)
model_calculator.add_as_observer_to_database(database)

acquisitor = LowerConfidenceBoundAcquisitor(model_calculator, kappa=KAPPA, verbose=True)

################################################################################################
# Generator settings:
################################################################################################

fractions = np.array([0.7, 0.7, 0])
confinement_cell = template.get_cell() * fractions.T
cell_corner = template.get_cell() @ np.array([0.15, 0.15, 0])
cell_corner[2] = np.max(template.positions[:, 2])-1
confinement_cell[2, 2] = 10

BC = BoxConstraint(confinement_cell, cell_corner, indices=environment.get_missing_indices())

start_generator = StartGenerator(confinement_cell=confinement_cell, cell_corner=cell_corner, may_nucleate_at_several_places=True)
rattle_generator = RattleGenerator(confinement_cell=confinement_cell, cell_corner=cell_corner)
replace_generator = CenterOfGeometryGenerator(confinement_cell=confinement_cell, cell_corner=cell_corner)

generators = [start_generator, rattle_generator, replace_generator]
num_samples = {0:[10, 15, 5]}
               
# ################################################################################################
# # Ensemble / Sampler / Acquisitor
# ################################################################################################

relaxer = PrimaryPostProcessIORelax(model=acquisitor.get_acquisition_calculator(), database=database, start_relax=10, sleep_timing=0.2, 
                                        optimizer_run_kwargs={'fmax':0.2, 'steps':50}, model_training_mode='primary', 
                                        optimizer_kwargs={'logfile':None}, 
                                        optimizer='BFGS', constraints=[BC])
                                        
postprocessors = [relaxer]

# Ensemble 
sampler = SamplerKMeans(feature_calc, database=database, sample_size=10, max_energy=25, use_saved_features=True)

collector = TimeDependentCollector(generators=generators, sampler=sampler, environment=environment, num_samples=num_samples, report_timing=True)

################################################################################################
# Let get the show running! 
################################################################################################

agox = AGOX(environment=environment, database=database, collector=collector, sampler=sampler, 
            acquisitor=acquisitor, seed=run_idx, postprocessors=postprocessors, evalulator=evaluator)

agox.run(N_episodes=NUM_EPISODES)

