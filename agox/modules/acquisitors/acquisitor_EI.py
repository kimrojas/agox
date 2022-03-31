import numpy as np
from agox.modules.acquisitors.acquisitor_ABC import AcquisitorBaseClass, AcquisitonCalculatorBaseClass
from ase.calculators.calculator import all_changes
from scipy.stats import norm

class ExpectedImprovementAcquisitor(AcquisitorBaseClass):

    name = 'ExpectedImprovement'

    def __init__(self, model, database, epsilon=0.01, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.model = model
        self.database = database

    def acquisition_function(self, E, sigma, Emin):
        Z = (Emin - E - self.epsilon)/sigma
        EI = (Emin - E - self.epsilon) * norm.cdf(Z) + sigma*norm.pdf(Z)
        return EI

    def acquisition_derivative(self, E, F, sigma, sigma_force, Emin):
        raise NotImplementedError('Acquisition Derivative Not Implemented')

    def calculate_acquisition_function(self, candidates):
        expected_improvement = np.zeros(len(candidates))

        if self.model.ready_state:
            best_energy = np.min([cand.get_potential_energy() for cand in self.database.get_all_candidates()])
            for i, candidate in enumerate(candidates):
                candidate.set_calculator(self.model)            
                E = candidate.get_potential_energy()
                sigma = candidate.get_uncertainty()
                expected_improvement[i] = self.acquisition_function(E, sigma, best_energy)

                candidate.add_meta_information('model_energy', E)
                candidate.add_meta_information('uncertainty', sigma)
                candidate.add_meta_information('expected_improvement', expected_improvement[i])

        # We're going to select according to argmin of fitness but EI is a positive quantity:
        expected_improvement = -expected_improvement
        return expected_improvement

    def print_information(self, candidates, acquisition_values):
        if self.model.ready_state:
            for i, candidate in enumerate(candidates):
                Emodel = candidate.get_meta_information('model_energy')
                sigma = candidate.get_meta_information('uncertainty')
                expected_improvement = candidate.get_meta_information('expected_improvement')
                print('Candidate: E = {:6.3f}, s = {:6.3f}, EI = {:6.3f}'.format(Emodel, sigma, expected_improvement))

    def assign_from_main(self, main):
        super().assign_from_main(main)
        self.model.assign_from_main(main)

    def get_acquisition_calculator(self):
        return ExpectedImprovementCalculator(self.model, self.acquisition_function, self.acquisition_derivative)

class ExpectedImprovementCalculator(AcquisitonCalculatorBaseClass):

    implemented_properties = ['energy', 'forces']

    def __init__(self, model_calculator, acquisition_function, acquisition_derivative, **kwargs):
        super().__init__(model_calculator, **kwargs)
        self.acquisition_function = acquisition_function
        self.acquisition_derivative = acquisition_derivative
        self.numerical_indices = None

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if 'forces' in properties:
            E, sigma = self.model_calculator.predict_energy(atoms, return_uncertainty=True)
            F, sigma_force = self.model_calculator.predict_forces(atoms, return_uncertainty=True, acquisition_function=self.acquisition_function)
            #self.results['forces'] = self.acquisition_derivative(E, F, sigma, sigma_force)
            self.results['forces'] = self.calculate_numerical_forces(atoms)
        else:
            E, sigma = self.model_calculator.predict_energy(atoms, return_uncertainty=True)
            
        self.results['energy'] = self.acquisition_function(E, sigma, self.get_best_energy())

    def calculate_numerical_forces(self, atoms, d=0.001):
        Emin = self.get_best_energy()
        F = np.zeros((len(atoms), 3))
        p0 = atoms.positions.copy()
        temp = atoms.copy()
        for a in self.numerical_indices:
            temp.set_positions(p0)
            for i in range(3):
                temp.positions[a, i] += d
                Ep, sp = self.model_calculator.predict_energy(temp, return_uncertainty=True)
                ap = self.acquisition_function(Ep, sp, Emin)
                temp.positions[a, i] -= 2*d
                Em, sm = self.model_calculator.predict_energy(temp, return_uncertainty=True)
                am = self.acquisition_function(Em, sm, Emin)
                F[a, i] = (am - ap) / (2 * d)

        # We want to maximize EI so: 
        F = -F

        return F

    def get_model_parameters(self):
        parameters = super().get_model_parameters()
        parameters['Ebest'] = self.get_best_energy_from_database()
        return parameters

    def set_model_parameters(self, parameters):
        super().set_model_parameters(parameters)
        self.best_energy = parameters['Ebest']
    
    def assign_from_main(self, main):
        super().assign_from_main(main)
        self.get_best_energy_from_database = main.database.get_best_energy

    def get_best_energy(self):
        if hasattr(self, 'get_best_energy_from_database'):
            return self.get_best_energy_from_database()
        else:
            return self.best_energy