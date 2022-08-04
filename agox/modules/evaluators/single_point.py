from .ABC_evaluator import EvaluatorBaseClass
import numpy as np
from copy import deepcopy
from ase.calculators.singlepoint import SinglePointCalculator

from timeit import default_timer as dt

class SinglePointEvaluator(EvaluatorBaseClass):

    name = 'EnergyEvaluator'

    def __init__(self, calculator, dummy_mode=False, **kwargs): 
        super().__init__(**kwargs)
        self.calculator = calculator
        self.dummy_mode = dummy_mode

    def evaluate_candidate(self, candidate):
        t0 = dt()
        candidate.set_calculator(self.calculator)
        
        if not self.dummy_mode: 
            try:
                E = candidate.get_potential_energy()
                F = candidate.get_forces()
                candidate.add_meta_information('SPC', 1)
            except Exception as e:
                self.writer('Energy calculation failed with exception: {}'.format(e))
                return False
        else:
            E = 0
            F = np.zeros((len(candidate), 2))
            
        calc = SinglePointCalculator(candidate, energy=E, forces=F)
        candidate.set_calculator(calc)

        return True