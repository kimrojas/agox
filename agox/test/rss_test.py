import pytest 
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
from agox import AGOX
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator
from ase import Atoms

def test_rss(tmp_path):

    d = tmp_path / "rss"
    d.mkdir()
    os.chdir(d)

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
    db_path = 'db{}.db'.format(0) # From input argument!
    database = Database(filename=db_path, order=3)

    ##############################################################################
    # Search Settings:
    ##############################################################################
        
    random_generator = RandomGenerator(**environment.get_confinement(), 
        environment=environment, order=1)

    evaluator = LocalOptimizationEvaluator(calc, gets={'get_key':'candidates'}, 
        optimizer_run_kwargs={'fmax':0.05, 'steps':5}, store_trajectory=False,
        order=2, constraints=environment.get_constraints())

    ##############################################################################
    # Let get the show running! 
    ##############################################################################
        
    agox = AGOX(random_generator, database, evaluator)

    agox.run(N_iterations=10)