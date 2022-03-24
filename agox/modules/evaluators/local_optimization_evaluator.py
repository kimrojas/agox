from re import A
from .evaluator_ABC import EvaluatorBaseClass
import numpy as np
from copy import deepcopy
from ase.calculators.singlepoint import SinglePointCalculator
from timeit import default_timer as dt

from ase.optimize.bfgs import BFGS
from ase.constraints import FixAtoms

class LocalOptimizationEvaluator(EvaluatorBaseClass):

    name = 'LocalOptimizationEvaluator'

    def __init__(self, calculator, optimizer=BFGS, optimizer_run_kwargs={'fmax':0.25, 'steps':200}, optimizer_kwargs={'logfile':None}, verbose=False, fix_template=True, constraints=[], dummy_mode=False, **kwargs): 
        super().__init__(**kwargs)
        self.calculator = calculator
        self.verbose = verbose
        self.dummy_mode = dummy_mode

        # Optimizer stuff:
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_run_kwargs = optimizer_run_kwargs

        # Constraints:
        self.constraints = constraints
        self.fix_template = fix_template

    def evaluate_candidate(self, candidate):
        t0 = dt()
        candidate.set_calculator(self.calculator)
        
        if not self.dummy_mode: 
            try:
                
                self.apply_constraints(candidate)

                optimizer = self.optimizer(candidate, **self.optimizer_kwargs)
                optimizer.run(**self.optimizer_run_kwargs)                
                candidate.add_meta_information('optimizer_steps', optimizer.get_number_of_steps())

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

    def apply_constraints(self, candidate):
        constraints = [] + self.constraints
        if self.fix_template:
            constraints.append(self.get_template_constraint(candidate))

        for constraint in constraints:
            if hasattr(constraint, 'reset'):
                constraint.reset()

        candidate.set_constraint(constraints)

    def get_template_constraint(self, candidate):
        return FixAtoms(indices=np.arange(len(candidate.template)))
