from agox.models.ABC_model import ModelBaseClass
import numpy as np
from agox.observer import Observer
from agox.writer import agox_writer
from ase.calculators.calculator import Calculator, all_changes
from agox.module import register_modules

class Ensemble(ModelBaseClass):

    name = 'Ensemble'
    implemented_properties = ['energy', 'forces', 'uncertainty', 'force_uncertainty']

    def __init__(self, models=None, **kwargs):
        super().__init__(**kwargs)
        self.n_models = len(models)
        register_modules(self, models, name='model')

    ############################################################################
    # Ensemble methods for training and predicting
    ############################################################################

    @property
    def models(self):
        return [getattr(self, f'model_{i}') for i in range(self.n_models)]

    def predict_energy(self, atoms=None, X=None, return_uncertainty=False, **kwargs):

        energies = self.predict_ensemble_energies(atoms=atoms)

        prediction = np.mean(energies)
        if return_uncertainty:
            return prediction, np.std(energies)
        return prediction
    
    def predict_forces(self, atoms=None, X=None, return_uncertainty=False, **kwargs):

        forces = self.predict_ensemble_forces(atoms=atoms)
        
        prediction = np.mean(forces, axis=0)
        if return_uncertainty:
            return prediction, np.std(forces, axis=0)
        return prediction

    def train_model(self, training_data, **kwargs):
        for model in self.models:
            model.train_model(training_data, **kwargs)
        
        self.ready_state = True

    ############################################################################
    # Methods for returning predictions of all models:
    ############################################################################

    def predict_ensemble_energies(self, atoms=None, X=None):
        return np.array([model.predict_energy(atoms=atoms, return_uncertainty=False) for model in self.models])
    
    def predict_ensemble_forces(self, atoms=None, X=None):
        return np.array([model.predict_forces(atoms=atoms, return_uncertainty=False) for model in self.models])

    # def predict_ensemble_forces(self, atoms=None, X=None):
    #     """
    #     Calculate numerical forces using the central difference method.
    #     """
    #     #return np.array([model.predict_forces(atoms=atoms, return_uncertainty=False) for model in self.model_list])

    @agox_writer
    @Observer.observer_method        
    def training_observer(self, database, state):
        """Observer method for use with on-the-fly training based data in an AGOX database.
        
        Note
        ----------
        This implementation simply calls the train_model method with all data in the database
        
        Parameters
        ----------
        atoms : AGOX Database object
            The database to keep the model trained against

        Returns
        ----------
        None
            
        """
        iteration = self.get_iteration_counter()

        if iteration < self.iteration_start_training:
            return
        if iteration % self.update_period != 0 and iteration != self.iteration_start_training:
            return

        data = database.get_all_candidates()
        self.train_model(data)

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        E, E_err = self.predict_energy(self.atoms, return_uncertainty=True)
        self.results['energy'] = E
        self.results['uncertainty'] = E_err
        
        if 'forces' in properties:
            forces,f_err = self.predict_forces(self.atoms, return_uncertainty=True)
            self.results['forces'] = forces
            self.results['force_uncertainty'] = f_err