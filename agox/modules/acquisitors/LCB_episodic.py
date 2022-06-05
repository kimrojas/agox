import numpy as np
from agox.modules.acquisitors.ABC_acquisitor import AcquisitonCalculatorBaseClass
from agox.modules.acquisitors import LowerConfidenceBoundAcquisitor
from ase.calculators.calculator import all_changes

class EpisodicLowerConfidenceBoundAcquisitor(LowerConfidenceBoundAcquisitor):

    name = 'EpisodicLowerConfindenceBoundAcquisitor'

    def __init__(self, model_calculator, func=None, **kwargs):
        super().__init__(model_calculator, **kwargs)

        if func is None:
            self.func = np.sqrt
        else: 
            self.func = func

    def calculate_acquisition_function(self, candidates):
        fitness = np.zeros(len(candidates))
        
        if self.model_calculator.ready_state:
            # Attach calculator and get model_energy
            for i, candidate in enumerate(candidates):
                candidate.set_calculator(self.model_calculator)
                E = candidate.get_potential_energy()
                sigma = candidate.get_uncertainty()
                fitness[i] = self.acquisition_function(E, sigma, self.get_iteration_counter())

                # For printing:
                candidate.add_meta_information('model_energy', E)
                candidate.add_meta_information('uncertainty', sigma)


        return fitness

    def acquisition_function(self, E, sigma, iteration):
        return E - self.iteration_kappa(iteration) * sigma

    def acquisition_force(self, E, F, sigma, sigma_force, iteration):
        return F - self.iteration_kappa(iteration) * sigma_force

    def iteration_kappa(self, iteration):
        return self.kappa * self.func(iteration)

    def get_acquisition_calculator(self, database):
        return EpsiodidcLowerConfidenceBoundCalculator(database, self.model_calculator, self.acquisition_function, self.acquisition_force)

class EpsiodidcLowerConfidenceBoundCalculator(AcquisitonCalculatorBaseClass):

    implemented_properties = ['energy', 'forces']

    def __init__(self, database, model_calculator, acquisition_function, acquisition_force, **kwargs):
        super().__init__(database, model_calculator, **kwargs)
        self.acquisition_function = acquisition_function
        self.acquisition_force = acquisition_force

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        iteration = self.get_iteration_number()

        if 'forces' in properties:
            E, sigma = self.model_calculator.predict_energy(atoms, return_uncertainty=True)
            F, sigma_force = self.model_calculator.predict_forces(atoms, return_uncertainty=True, acquisition_function=self.acquisition_function)
            self.results['forces'] = self.acquisition_force(E, F, sigma, sigma_force, iteration)
        else:
            E, sigma = self.model_calculator.predict_energy(atoms, return_uncertainty=True)
            
        self.results['energy'] = self.acquisition_function(E, sigma, iteration)



