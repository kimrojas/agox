import numpy as np
from agox.observer import Observer
from agox.writer import Writer, agox_writer
from ase.calculators.singlepoint import SinglePointCalculator

class ModelTrackerBeforeTraining(Observer, Writer):

    name = 'ModelTrackerBeforeTraining'

    def __init__(self, model, sets={'set_key':'model_tracker_candidates'}, gets={'get_key':'evaluated_candidates'}, order=5.5):
        Observer.__init__(self, sets=sets, gets=gets, order=order)
        Writer.__init__(self)
        self.model = model
        self.add_observer_method(self.cache_old_model_energies, sets=self.sets[0], gets=self.gets[0], order=self.order[0])

    @agox_writer
    @Observer.observer_method
    def cache_old_model_energies(self, state):

        if not self.model.ready_state:
            return 
        
        self.writer('Doing stuff')

        candidates_with_model_energies = []
        candidates = state.get_from_cache(self, self.get_key)
        for i,cached_candidate in enumerate(candidates):
            E_dft = cached_candidate.get_potential_energy()
            new_candidate = cached_candidate.copy()
            new_candidate.set_calculator(self.model)
            E_model = new_candidate.get_potential_energy()
            single_point_calc = SinglePointCalculator(new_candidate, energy=E_dft)
            new_candidate.set_calculator(single_point_calc)
            new_candidate.add_meta_information('model_energy',E_model)
            candidates_with_model_energies.append(new_candidate)

        state.add_to_cache(self, self.set_key, candidates_with_model_energies, mode='a')

            
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
            arrow = '<-- LARGE DELTA' if abs(Delta_old_E) or abs(Delta_new_E) > 0.5 else ''
                
            self.writer(f'TRACKER {i:04d} {E_dft:8.3f} {E_old_model:8.3f} {E_new_model:8.3f} {Delta_old_E:8.3f} {Delta_new_E:8.3f} {arrow:s}')


        




