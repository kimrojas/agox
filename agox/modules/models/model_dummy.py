import numpy as np
from .model_ABC import ModelBaseClass
from ase.calculators.calculator import Calculator, all_changes

class DummyModel(ModelBaseClass):

    implemented_properties = ['energy', 'uncertainty']

    name = 'DumbestModel'

    def __init__(self):
        super().__init__()
        self._ready_state = True

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.results['energy'] = 0
        self.results['uncertainty'] = 0

    def predict_energy(self, atoms, return_uncertainty=False, feature=None, **kwargs):
        if return_uncertainty:
            return 0, 0
        else:
            return 0

    def predict_forces(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        if return_uncertainty:
            return np.zeros((len(atoms), 3)), np.zeros((len(atoms), 3))
        else:
            return 0

    def train_model(self, training_data, energies):
        pass

    def get_model_parameters(self):
        parameters = {}
        return parameters
    
    def set_model_parameters(self, parameters):
        pass