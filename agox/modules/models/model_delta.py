from .model_ABC import ModelBaseClass
from ase.calculators.calculator import Calculator, all_changes

import numpy as np

class DeltaModel(ModelBaseClass):

    name = 'Delta'

    implemented_properties = ['energy', 'forces', 'uncertainty']

    def __init__(self, models, episode_start_training=10, update_interval=1, **kwargs):
        super().__init__(**kwargs)
        self.models = models

        self.episode_start_training = episode_start_training
        self.update_interval = update_interval

        self.has_pretraining_data = False

    def calculate(self, atoms, properties, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        E = 0; F = np.zeros((len(atoms), 3)); s = 0

        E, s = self.predict_energy(atoms, return_uncertainty=True)        
        self.results['energy'] = E
        self.results['uncertainty'] = s

        if 'forces' in properties:
            F = self.predict_forces(atoms)
            self.results['forces'] = F

    def predict_energy(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        """
        Main method for energy prediction. Always include **kwargs when implementing this function. 
        """
        E = 0; s = 0
        for model in self.models:
            if model.ready_state:
                dE, ds= model.predict_energy(atoms, return_uncertainty=True)
                E += dE; s += ds

        if return_uncertainty:
            return E, s
        else:
            return E

    def predict_forces(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        """
        Main method for force prediction. Always include **kwargs when implementing this function. 
        """
        F = np.zeros((len(atoms), 3))
        Fs = np.zeros((len(atoms), 3))

        for model in self.models:
            if model.ready_state:
                dF, dFs = model.predict_force(atoms, return_uncertainty=True)
                F += dF; Fs += dFs

        if return_uncertainty:
            return F, dFs
        else:
            return F

    def train_model(self, training_data, energies):
        """
        Delta-Learning training procedure:
        """
        #energies = [atoms.get_potential_energy() for atoms in training_data]

        delta_target = energies.copy()
        for i, model in enumerate(self.models):

            # Train the model: 
            model.train_model(training_data, delta_target)

            # Calculate the delta:
            if i+1 < len(self.models):
                E = model.batch_predict(training_data)
                delta_target = delta_target - E


        self.set_ready_state(True)

    def assign_from_main(self, main):
        super().assign_from_main(main)
        self.get_episode_counter = main.get_episode_counter

        for model in self.models:
            model.part_of_delta_model = True
            model.assign_from_main(main)

        main.database.attach_observer(self.name, self.training_observer_func)

    def get_name(self):
        name = ''
        for model in self.models:
            name += model.name
        return name

    def set_pretraining_data(self, data):
        self.pretraining_data = data
        self.has_pretraining_data = True
    
    def get_pretraining_data(self):
        return self.pretraining_data

    def training_observer_func(self, database):
        episode = self.get_episode_counter()
        if episode < self.episode_start_training:
            return
        if (episode % self.update_interval != 0) * (episode != self.episode_start_training):
            return 

        # Find complete builds:
        all_data = database.get_all_candidates()

        if self.has_pretraining_data:
            pre_data = self.get_pretraining_data()
            print('Pre-training data', len(pre_data))

            all_data += pre_data

        print('All data: {}'.format(len(all_data)))
        energies = np.array([x.get_potential_energy() for x in all_data])

        self.train_model(all_data, energies)

    def reset(self):
        super().reset()
        for model in self.models:
            model.reset()

    def get_model_parameters(self):
        parameters = {}
        for model in self.models:
            parameters[model.name] = model.get_model_parameters()

        return parameters
    
    def set_model_parameters(self, parameters):
        for model in self.models:
            model.set_model_parameters(parameters[model.name])

        self.set_ready_state(True)