import numpy as np
from agox.acquisitors.ABC_acquisitor import AcquisitorBaseClass, AcquisitonCalculatorBaseClass
from ase.calculators.calculator import all_changes

class BoundedLowerConfidenceBoundAcquisitor(AcquisitorBaseClass):

    name = 'LCBAcquisitor'

    def __init__(self, model_calculator, kappa=1, sigma_max=1, soft_cutoff=0.75, **kwargs):
        super().__init__(**kwargs)
        self.kappa = kappa
        self.model_calculator = model_calculator

        self.sigma_max = sigma_max
        self.soft_cutoff = soft_cutoff

    def calculate_acquisition_function(self, candidates):
        fitness = np.zeros(len(candidates))
        
        # Attach calculator and get model_energy
        for i, candidate in enumerate(candidates):
            candidate.set_calculator(self.model_calculator)
            E = candidate.get_potential_energy()
            sigma = candidate.get_uncertainty()
            fitness[i] = self.acquisition_function(E, sigma)

            # For printing:
            candidate.add_meta_information('model_energy', E)
            candidate.add_meta_information('uncertainty', sigma)

        return fitness

    def print_information(self, candidates, acquisition_values):
        if self.model_calculator.ready_state:
            for i, candidate in enumerate(candidates):
                fitness = acquisition_values[i]
                Emodel = candidate.get_meta_information('model_energy')
                sigma = candidate.get_meta_information('uncertainty')
                self.writer('Candidate: E={:8.3f}, s={:8.3f}, F={:8.3f}'.format(Emodel, sigma, fitness))

    def get_acquisition_calculator(self):
        return BoundedLowerConfidenceBoundCalculator(self.model_calculator, self.acquisition_function)
    
    @staticmethod
    def cutoff(sigma, sigma_c, a=0.75):
        if sigma < a*sigma_c:
            return sigma
        elif sigma >= a*sigma_c:
            return a * sigma_c + (sigma_c - a * sigma_c) * np.tanh((sigma - a*sigma_c)/(sigma_c - a*sigma_c))

    def acquisition_function(self, E, sigma):
        sigma_eff = self.cutoff(sigma, self.sigma_max, self.soft_cutoff)
        return E - self.kappa * sigma_eff

    def do_check(self, **kwargs):
        return self.model_calculator.ready_state

class BoundedLowerConfidenceBoundCalculator(AcquisitonCalculatorBaseClass):

    implemented_properties = ['energy', 'forces']

    def __init__(self, model_calculator, acquisition_function, **kwargs):
        super().__init__(model_calculator, **kwargs)
        self.acquisition_function = acquisition_function

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if 'forces' in properties:
            E, sigma = self.model_calculator.predict_energy(atoms, return_uncertainty=True)
            self.results['forces'] = self.central_difference_force(atoms)
        else:
            E, sigma = self.model_calculator.predict_energy(atoms, return_uncertainty=True)
            
        self.results['energy'] = self.acquisition_function(E, sigma)

    def central_difference_force(self, atoms, d=0.001):
        F = np.zeros((len(atoms), 3))
        atoms = atoms.copy()
        new_positions = atoms.positions.copy()
        for i in range(len(atoms)):
            for j in range(3):
                new_positions[i, j] += d
                atoms.set_positions(new_positions)
                E1, sigma1 = self.model_calculator.predict_energy(atoms, return_uncertainty=True)
                L1 = self.acquisition_function(E1, sigma1)
                new_positions[i, j] -= 2 * d
                atoms.set_positions(new_positions)
                E2, sigma2 = self.model_calculator.predict_energy(atoms, return_uncertainty=True)
                L2 = self.acquisition_function(E2, sigma2)
                new_positions[i, j] += d
                atoms.set_positions(new_positions)
                F[i, j] = -(L1 - L2) / (2 * d)
        return F






