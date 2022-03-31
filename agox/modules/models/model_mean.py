import numpy as np
from .model_ABC import ModelBaseClass
from ase.calculators.calculator import Calculator, all_changes

class MeanModel(ModelBaseClass):

    implemented_properties = ['energy', 'forces']

    name = 'Mean' #'MeanModelNotNice'

    mean = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train_model(self, training_data, energies):                
        self.mean = np.mean(energies)
        self.set_ready_state(True)


    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.results['energy'] = self.mean
        self.results['forces'] = np.zeros((len(atoms), 3))

    def batch_predict(self, data, over_write=False):
        E = np.ones(len(data)) * self.mean
        if over_write:
            self.batch_assign(data, E)            
        return E

    def predict_energy(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        if return_uncertainty:
            return self.mean, 0
        else:
            return self.mean

    def predict_forces(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        if return_uncertainty:
            return np.zeros((len(atoms), 3)), np.zeros((len(atoms), 3))
        else:
            return np.zeros((len(atoms), 3))

    def get_model_parameters(self):
        parameters = {}
        parameters['mean'] = self.mean
        return parameters
    
    def set_model_parameters(self, parameters):
        self.mean = parameters['mean']
