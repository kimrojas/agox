from re import A
from .ABC_evaluator import EvaluatorBaseClass
import numpy as np
from copy import deepcopy
from ase.calculators.singlepoint import SinglePointCalculator
from timeit import default_timer as dt

from ase.optimize.bfgs import BFGS
from ase.constraints import FixAtoms
from ase.io import read

class LocalOptimizationEvaluator(EvaluatorBaseClass):

    name = 'LocalOptimizationEvaluator'

    def __init__(self, calculator, optimizer=BFGS, optimizer_run_kwargs={'fmax':0.25, 'steps':200}, optimizer_kwargs={'logfile':None}, verbose=False, 
        fix_template=True, constraints=[], dummy_mode=False, use_all_traj_info=True, **kwargs): 
        super().__init__(**kwargs)
        self.calculator = calculator
        self.verbose = verbose
        self.dummy_mode = dummy_mode

        # Optimizer stuff:
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_run_kwargs = optimizer_run_kwargs
        self.use_all_traj_info = use_all_traj_info
        if use_all_traj_info and not 'trajectory' in optimizer_kwargs:
            self.optimizer_kwargs['trajectory'] = 'tmp.traj'

        # Constraints:
        self.constraints = constraints
        self.fix_template = fix_template

    def evaluated_candidates_append(self, candidate):
        if self.use_all_traj_info:
            try:
                # read the DFT calculations along the trajectory and add them
                traj = read(self.optimizer_kwargs['trajectory'],index=':-1')
                for t in traj:
                    E = t.get_potential_energy()
                    F = t.get_forces()
                    candidate_along_trajectory = candidate.copy()
                    candidate_along_trajectory.set_positions(t.get_positions())
                    calc = SinglePointCalculator(candidate_along_trajectory, energy=E, forces=F)
                    candidate_along_trajectory.set_calculator(calc)

                    candidate.add_meta_information('final', False)
                    self.evaluated_candidates.append(candidate_along_trajectory)
            except:
                pass

        # add the final candidate
        candidate.add_meta_information('final', True)
        self.evaluated_candidates.append(candidate)
        

    def evaluate_candidate(self, candidate):
        t0 = dt()
        candidate.set_calculator(self.calculator)
        
        if not self.dummy_mode: 
            try:
                
                self.apply_constraints(candidate)

                optimizer = self.optimizer(candidate, **self.optimizer_kwargs)
                optimizer.run(**self.optimizer_run_kwargs)                
                candidate.add_meta_information('SPC', optimizer.get_number_of_steps()+1)

                E = candidate.get_potential_energy()
                F = candidate.get_forces()
            except Exception as e:
                self.writer('Energy calculation failed with exception: {}'.format(e))
                return False
        else:
            E = 0
            F = np.zeros((len(candidate), 2))
            
        calc = SinglePointCalculator(candidate, energy=E, forces=F)
        candidate.set_calculator(calc)

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
