import matplotlib

matplotlib.use('Agg')
import argparse
import numpy as np
from ase.io import read

# Common AGOX
from agox import AGOX
from agox.modules.databases import Database
from agox.modules.environments.environment_singular import EnvironmentSingular
from agox.modules.acquisitors.acquisitor_LCB import LowerConfidenceBoundAcquisitor
from agox.modules.models.model_GPR import ModelGPR
from agox.modules.models.gaussian_process.default_model_gofee import get_default_GPR_model
from agox.modules.evaluators.energy_evaluator import EnergyEvaluator
from agox.modules.collectors import TimeDependentCollector
from agox.modules.generators import RattleGenerator, StartGenerator

from agox.modules.postprocessors.postprocess_io_relax import PrimaryPostProcessIORelax
from agox.modules.samplers.sampler_kmeans import SamplerKMeans
from agox.modules.postprocessors.constraints.box_constraint import BoxConstraint

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

from agox.modules.helpers.gpaw_io import GPAW_IO

calc = GPAW_IO(mode='PW(300)',
            xc='PBE',
            basis='dzp',
            maxiter='200',
            kpts ='(1, 1, 1)',
            convergence="{'energy':0.005, 'density':1.0e-3, 'eigenstates':1.0e-3, 'bands':'occupied'}",
            occupations="FermiDirac(0.1)",
            gpts = "h2gpts(0.2, t.get_cell(), idiv = 8)",
            nbands='110%',
            txt='dft_log_PW.txt', 
            modules=['from gpaw.utilities import h2gpts', 'from gpaw import FermiDirac'])

evalulator = EnergyEvaluator(calc, verbose=True)

################################################################################################
# General settings:
################################################################################################

template = read('/home/machri/Projects/agox/runscripts/agox_prod_2021/Ru3N4_graphene/template/paper_matching_template_4C.traj')
environment = EnvironmentSingular(template=template, symbols='Ru3N4C4')

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
model_calculator = ModelGPR(model, database=database, update_interval=1, optimize_frequency=25, optimize_loglikelyhood=True, use_saved_features=True)

acquisitor = LowerConfidenceBoundAcquisitor(model_calculator, kappa=2)

################################################################################################
# Generator settings:
################################################################################################
template_0C = read('/home/machri/Projects/agox/runscripts/agox_prod_2021/Ru3N4_graphene/template/paper_matching_template.traj')
cell_corner = template_0C[32].position.copy()
cell_corner[2] = 4
confinement_cell = np.array([
        [template_0C[26].x-template_0C[32].x, 0, 0],
        [0, template_0C[41].y-template_0C[32].y, 0],
        [0, 0, 8]
        ])
confinement_cell[0, 0] += 2.5
confinement_cell[1, 1] += 2.5
cell_corner[0:2] -= 1.25

BC = BoxConstraint(confinement_cell, cell_corner, indices=environment.get_missing_indices())

start_generator = StartGenerator(confinement_cell=confinement_cell, cell_corner=cell_corner, may_nucleate_at_several_places=True)
rattle_generator = RattleGenerator(confinement_cell=confinement_cell, cell_corner=cell_corner)

generators = [start_generator, rattle_generator]
num_samples = {0:[10, 20]}
               
# ################################################################################################
# # Ensemble / Sampler / Acquisitor
# ################################################################################################

relaxer = PrimaryPostProcessIORelax(model=acquisitor.get_acquisition_calculator(), database=database, 
                                        start_relax=10, sleep_timing=0.2, 
                                        optimizer_run_kwargs={'fmax':0.2, 'steps':50}, 
                                        optimizer_kwargs={'logfile':None}, 
                                        optimizer='BFGS', constraints=[BC])

postprocessors = [relaxer]

# Ensemble 
sampler = SamplerKMeans(feature_calc, database=database, sample_size=10, max_energy=25, use_saved_features=True)

collector = TimeDependentCollector(generators=generators, sampler=sampler, environment=environment, 
                                   num_samples=num_samples, report_timing=True)

################################################################################################
# Let get the show running! 
################################################################################################

agox = AGOX(environment=environment, database=database, collector=collector, sampler=sampler, 
            acquisitor=acquisitor, evalulator=evalulator, seed=run_idx)

agox.run(N_episodes=10)

