from abc import ABC, abstractmethod
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from agox.observer_handler import Observer, ObserverHandler

import numpy as np

class ModelBaseClass(Calculator, ABC, Observer):

    def __init__(self, verbose=True, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self._ready_state = False
        self.part_of_delta_model = False

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def train_model(self, training_data, energies):
        """
        If your model does not need to train then just write a method that does nothing, the reason 
        for this required method is to ensure that methods that can train take training data in 
        the same way. 

        Training_data: List of Atoms-objects.
        energies: List/array of target energies. 

        While the training_data MAY have attached energies that can be accesed using .get_potential_energy()
        the training should NOT use those but rely on the energies array. 
        """
        pass

    def training_observer_func(self, database):
        """
        Function called by database that starts the training. 
        """
        pass

    @property
    def ready_state(self):
        return self._ready_state

    @property
    def ever_trained(self):
        """
        Makes it easier to transition to new code.
        """
        return self._ready_state

    @abstractmethod
    def predict_energy(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        """
        Main method for energy prediction. Always include **kwargs when implementing this function. 
        """
        pass

    @abstractmethod
    def predict_forces(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        """
        Main method for force prediction. Always include **kwargs when implementing this function. 
        """
        pass

    def set_ready_state(self, state):
        self._ready_state = bool(state)
    
    def assign_from_main(self, main):
        super().assign_from_main(main)

    def batch_predict(self, data, over_write=False):
        """
        Predict the energy of several structures WITHOUT overwriting their .get_potential_energy()

        Overwrite this method if your Model has some better of doing this. 
        """

        E = np.zeros(len(data))
        if not over_write:
            for i, atoms in enumerate(data):
                atoms_ = atoms.copy()
                atoms_.set_calculator(self)
                E[i] = atoms_.get_potential_energy()
        elif over_write:
            for i, atoms in enumerate(data):
                atoms.set_calculator(self)
                E[i] = atoms.get_potential_energy()
        return E

    def batch_assign(self, data, energy):
        """
        Attaches single-point calculator to atoms in data with the energies in energy.

        Can be used to easily implement batch_predict with over_write = True.
        """ 
        for atoms, energy in zip(data, energy):
            calc = SinglePointCalculator(atoms, energy=energy)
            atoms.set_calc(calc)

    def get_uncertainty(self, atoms):
        if 'uncertainty' in self.implemented_properties:
            return self.get_property('uncertainty', atoms)
        else:
            return 0

    def set_verbosity(self, verbose):
        self.verbose = verbose

    def get_model_parameters(self, *args, **kwargs):
        raise NotImplementedError('''get_model_parameters has not been implemeneted for this type of model. Do so if you need 
                            functionality that relies on this method''')

    def set_model_parameters(self, *args, **kwargs):
        raise NotImplementedError('''set_model_parameters has not been implemeneted for this type of model. Do so if you need 
                            functionality that relies on this method''')

    def remove_as_observer_to_database(self, database):
        database.delete_observer(self.training_observer_func)

    def add_as_observer_to_database(self, database):
        database.attach_observer(self.name, self.training_observer_func)