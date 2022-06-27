from re import A
from .ABC_evaluator import EvaluatorBaseClass
import numpy as np
from copy import deepcopy
from ase.calculators.singlepoint import SinglePointCalculator
from timeit import default_timer as dt
from agox.modules.helpers.writer import header_footer

from ase.optimize.bfgs import BFGS
from ase.constraints import FixAtoms
from ase.io import read

class LocalOptimizationEvaluator(EvaluatorBaseClass):

    name = 'LocalOptimizationEvaluator'

    def __init__(self, calculator, optimizer=BFGS, optimizer_run_kwargs={'fmax':0.25, 'steps':200},
                 optimizer_kwargs={'logfile':None}, verbose=False, fix_template=True, constraints=[],
                 store_trajectory=True, **kwargs): 
        super().__init__(**kwargs)
        self.calculator = calculator
        self.verbose = verbose
        self.store_trajectory = store_trajectory
        
        # Optimizer stuff:
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_run_kwargs = optimizer_run_kwargs

        # Constraints:
        self.constraints = constraints
        self.fix_template = fix_template


    @header_footer
    def evaluate_candidates(self):
        if not self.store_trajectory:
            super().evaluate_candidates()
        else:
            candidates = self.get_from_cache(self.get_key)
            done = False

            self.evaluated_candidates = []
            while candidates and not done:            

                candidate = candidates.pop(0)
                state = self.evaluate_candidate(candidate)

                if state:
                    self.evaluated_candidates.append(candidate)
                    if len(self.evaluated_candidates) == self.number_to_evaluate:
                        done = True

    def evaluate_candidate(self, candidate):
        candidate.set_calculator(self.calculator)
        self.apply_constraints(candidate)
        optimizer = self.optimizer(candidate, **self.optimizer_kwargs)
        if self.store_trajectory:
            optimizer.attach(self._observer, interval=1, candidate=candidate, steps=optimizer.get_number_of_steps)
        
        try:            
            optimizer.run(**self.optimizer_run_kwargs)                
            candidate.add_meta_information('relax_index', optimizer.get_number_of_steps())
            
        except Exception as e:
            self.writer('Energy calculation failed with exception: {}'.format(e))
            return False

        E = candidate.get_potential_energy()
        F = candidate.get_forces()
        self.writer(f'e={E}')
        calc = SinglePointCalculator(candidate, energy=E, forces=F)
        candidate.set_calculator(calc)

        return True


    def _observer(self, candidate, steps):
        E = candidate.get_potential_energy()
        F = candidate.get_forces()
        
        traj_candidate = candidate.copy()
        calc = SinglePointCalculator(traj_candidate, energy=E, forces=F)
        traj_candidate.set_calculator(calc)
        traj_candidate.add_meta_information('relax_index', steps())
        self.writer(f'energy: {E:.3f}, steps {steps()}')
        self.add_to_cache(self.set_key, [traj_candidate], mode='a')
    

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
