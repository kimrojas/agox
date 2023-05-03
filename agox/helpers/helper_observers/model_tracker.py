import numpy as np
import os
from agox.observer import Observer
from agox.writer import Writer, agox_writer
from ase.calculators.singlepoint import SinglePointCalculator

class ModelTrackerBeforeTraining(Observer, Writer):

    name = 'ModelTrackerBeforeTraining'

    def __init__(self, model, sets={'set_key':'model_tracker_candidates'}, gets={'get_key':'evaluated_candidates'}, order=5.5, 
                save_path='', save_name='', save=False):
        Observer.__init__(self, sets=sets, gets=gets, order=order)
        Writer.__init__(self)
        self.model = model
        self.add_observer_method(self.cache_old_model_energies, sets=self.sets[0], gets=self.gets[0], order=self.order[0])
        self.save = save
        self.save_path = save_path
        self.save_name = save_name

    @agox_writer
    @Observer.observer_method
    def cache_old_model_energies(self, state):

        if not self.model.ready_state:
            return 
        
        self.writer('Doing stuff')

        candidates_with_model_energies = []
        candidates = state.get_from_cache(self, self.get_key)
        for i, cached_candidate in enumerate(candidates):
            E_dft = cached_candidate.get_potential_energy()
            new_candidate = cached_candidate.copy()
            new_candidate.set_calculator(self.model)
            E_model = new_candidate.get_potential_energy()

            self.writer(f'{i}: {E_dft} - {E_model}')

            single_point_calc = SinglePointCalculator(new_candidate, energy=E_dft)
            new_candidate.set_calculator(single_point_calc)
            new_candidate.add_meta_information('model_energy',E_model)
            candidates_with_model_energies.append(new_candidate)

        state.add_to_cache(self, self.set_key, candidates_with_model_energies, mode='a')

        if self.save:
            self.save_structures(candidates_with_model_energies)
            self.save_model()
        
    def save_structures(self, candidates):
        from ase.io import write
        path = os.path.join(self.save_path, self.save_name+f'candidates_iteration{self.get_iteration_counter()}.traj')
        write(path, candidates)

    def save_model(self):
        self.model.save(directory=self.save_path, prefix=self.save_name+f'_model_iteration_{self.get_iteration_counter()}.traj')
            
class ModelTrackerAfterTraining(Observer, Writer):

    name = 'ModelTrackerAfterTraining'

    def __init__(self, model, sets={}, gets={'get_key':'model_tracker_candidates'}, order=6.5):
        Observer.__init__(self, sets=sets, gets=gets, order=order)
        Writer.__init__(self, verbose=True)
        self.model = model
        self.add_observer_method(self.dump_model_energies, sets=self.sets[0], gets=self.gets[0], order=self.order[0])

    @agox_writer
    @Observer.observer_method
    def dump_model_energies(self, state):

        if not self.model.ready_state:
            return 

        self.writer(f'TRACKER      {"DFT":8s} {"Epre":8s} {"Epost":8s} {"Epre-DFT":8s} {"Epost-DFT":8s}')
        candidates = state.get_from_cache(self, self.get_key)

        if candidates is None:
            return

        for i,candidate in enumerate(candidates):
            E_dft = candidate.get_potential_energy()
            E_old_model = candidate.get_meta_information('model_energy')
            candidate.set_calculator(self.model)
            E_new_model = candidate.get_potential_energy()
            Delta_old_E = E_old_model - E_dft
            Delta_new_E = E_new_model - E_dft
            arrow = '<-- LARGE DELTA' if abs(Delta_old_E) > 0.5 or abs(Delta_new_E) > 0.5 else ''
                
            self.writer(f'TRACKER {i:04d} {E_dft:8.3f} {E_old_model:8.3f} {E_new_model:8.3f} {Delta_old_E:8.3f} {Delta_new_E:8.3f} {arrow:s}')

class ModelValidator(Observer, Writer):

    name = 'ModelValidator'

    def __init__(self, model, validation_data, sets={}, gets={}, order=7.5):
        Observer.__init__(self, sets=sets, gets=gets, order=order)
        Writer.__init__(self, verbose=True)
        self.model = model
        self.add_observer_method(self.dump_model_energies, sets=self.sets[0], gets=self.gets[0], order=self.order[0])
        self.validation_data = validation_data
        self.save_path = ''
        self.save_name = 'validation'

    @agox_writer
    @Observer.observer_method
    def dump_model_energies(self, state):

        if not self.model.ready_state:
            return 

        self.writer(f'VALIDATOR      {"DFT":8s} {"Emodel":8s} {"Emodel-DFT":8s} {"sigma":8s}')

        energies = []
        sigmas = []
        for i,candidate in enumerate(self.validation_data):
            E_dft = candidate.get_potential_energy()
            E_model, sigma = self.model.predict_energy(candidate, return_uncertainty=True)
            energies.append(E_model)
            sigmas.append(sigma)

            Delta_E = E_model - E_dft
            arrow = '<-- LARGE DELTA' if abs(Delta_E) > 0.5 else ''
                
            self.writer(f'VALIDATOR {i:04d} {E_dft:8.3f} {E_model:8.3f} {Delta_E:8.3f} {sigma:8.3f} {arrow:s}')


        path = os.path.join(self.save_path, f'{self.save_name}_{self.get_iteration_counter():04d}.npy')
        with open(path, 'wb') as f:
            np.save(f, np.array(energies))
            np.save(f, np.array(sigmas))
        




