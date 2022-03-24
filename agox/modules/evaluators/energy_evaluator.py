from .evaluator_ABC import EvaluatorBaseClass
import numpy as np
from copy import deepcopy
from ase.calculators.singlepoint import SinglePointCalculator

from timeit import default_timer as dt

class EnergyEvaluator(EvaluatorBaseClass):

    name = 'EnergyEvaluator'

    def __init__(self, calculator, verbose=False, dummy_mode=False, **kwargs): 
        super().__init__(**kwargs)
        self.calculator = calculator
        self.verbose = verbose
        self.dummy_mode = dummy_mode

    def evaluate_candidate(self, candidate):
        t0 = dt()
        candidate.set_calculator(self.calculator)
        
        if not self.dummy_mode: 
            try:
                E = candidate.get_potential_energy()
                F = candidate.get_forces()
            except Exception as e:
                print('Energy calculation failed with exception: {}'.format(e))
                return False
        else:
            E = 0
            F = np.zeros((len(candidate), 2))
            
        calc = SinglePointCalculator(candidate, energy=E, forces=F)
        candidate.set_calculator(calc)
        
        if self.verbose:
            print('Energy calculation time: {}'.format(dt()-t0))

        return True