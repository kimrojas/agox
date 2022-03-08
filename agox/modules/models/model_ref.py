from .model_ABC import ModelBaseClass
import numpy as np

from ase.calculators.calculator import Calculator, all_changes

class ReferenceModel(ModelBaseClass):

    name = 'Ref'

    implemented_properties = ['energy', 'forces']

    def __init__(self, reference_dict={}, **kwargs):
        super().__init__(**kwargs)
        self.reference_dict = reference_dict
        self.set_ready_state(True)

    def predict_energy(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        numbers = atoms.get_atomic_numbers()
        energy = np.sum([self.reference_dict[number] for number in numbers])
        if return_uncertainty:
            return energy, 0
        else:
            return energy

    def predict_forces(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        if return_uncertainty:
            return np.zeros((len(atoms), 3)), np.zeros((len(atoms), 3))
        else:
            return np.zeros((len(atoms), 3))

    def calculate(self, atoms, properties, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)        
        self.results['energy'] = self.predict_energy(atoms)
        self.results['forces'] = self.predict_forces(atoms)

    def train_model(self, training_data, energies):
        pass

    def get_model_parameters(self):
        parameters = {}
        parameters['reference_dict'] = self.reference_dict
        return parameters

    def set_model_parameters(self, parameters):
        self.reference_dict = parameters['reference_dict']

