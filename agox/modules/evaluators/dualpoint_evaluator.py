from .evaluator_ABC import EvaluatorBaseClass
from copy import deepcopy
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from timeit import default_timer as dt

class DualPointEvaluator(EvaluatorBaseClass):
    gauge = 'DualPointEvaluator'
    def __init__(self, calculator, rmax=0.1, fmax=5, fix_template=True, constraints=[], verbose=False, number_to_gauge=1):
        super().__init__(number_to_gauge=2*number_to_gauge)
        self.calculator = calculator
        self.rmax = rmax
        self.fmax = fmax
        self.fix_template = fix_template
        self.constraints = constraints
        self.verbose = verbose

    def gauge_candidate(self, candidate):
        t0 = dt()
        candidate.set_calculator(deepcopy(self.calculator))
        candidate_step = candidate.copy()
        self.apply_constraints(candidate_step)
        candidate_step.set_calculator(deepcopy(self.calculator))
        try:
            E = candidate.get_potential_energy()
            F = candidate.get_forces()
        except Exception as e:
            print('Energy calculation failed with exception: {}'.format(e))
            return False

        calc = SinglePointCalculator(candidate, energy=E, forces=F)
        candidate.set_calculator(calc)
        
        f_flat = np.sqrt((F**2).sum(axis=1).max())
        pos_displace = self.rmax * F*min(1/self.fmax, 1/f_flat)
        pos_dp = candidate_step.positions + pos_displace
        candidate_step.set_positions(pos_dp)
        
        try:
            E = candidate_step.get_potential_energy()
            F = candidate_step.get_forces()
        except Exception as e:
            print('Dualpoint calculation failed with exception: {}'.format(e))
            return True

        calc = SinglePointCalculator(candidate_step, energy=E, forces=F)
        candidate_step.set_calculator(calc)
        self.remove_constraints(candidate_step)
        
        self.gauged_candidates.append(candidate_step)

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

    def remove_constraints(self, candidate):
        candidate.set_constraint([])

    def get_template_constraint(self, candidate):
        return FixAtoms(indices=np.arange(len(candidate.template)))



        
