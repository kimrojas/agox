import numpy as np 
from ase import Atoms
from .ABC_database import DatabaseBaseClass
from agox.modules.candidates import StandardCandidate
from copy import deepcopy

class MemoryDatabase(DatabaseBaseClass):
    """ Database module """
    
    name = 'MemoryDatabase'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.number_of_preset_candidates = 0

    ####################################################################################################################
    # Memory-based methods:
    ####################################################################################################################

    def store_candidate(self, candidate, accepted=True, write=True, dispatch=True):
        # Needs some way of handling a dummy candidate, probably boolean argument.
        if accepted:
            self.candidates.append(candidate)
            self.candidate_energies.append(candidate.get_potential_energy())
        if dispatch:
            self.dispatch_to_observers(self)

    def get_all_candidates(self, return_preset=True):
        min_index = self.get_minimum_index(return_preset)
        all_candidates = []
        for candidate in self.candidates[min_index:]:
            all_candidates.append(candidate)
        return all_candidates

    def get_most_recent_candidate(self):
        if len(self.candidates) > 0:
            candidate = self.candidates[-1]
        else:
            candidate = None
        return candidate

    def get_recent_candidates(self, number):
        return [candidate for candidate in self.candidates[-number:]]

    def get_best_energy(self, return_preset=True):
        min_index = self.get_minimum_index(return_preset)
        try:
            return np.min(self.candidate_energies[min_index:])
        except:
            return np.inf
        
    def assign_from_main(self, main):
        super().assign_from_main(main)
    
    def get_iteration_counter(self):
        """
        Overwritten when added as an observer.
        """
        return 0

    def write(self, *args, **kwargs):
        pass

    def reset(self):
        self.candidates = []
        self.candidate_energies = []

    def set_number_of_preset_candidates(self, number):
        self.number_of_preset_candidates = number
    
    def get_minimum_index(self, return_preset):
        if return_preset:
            min_index = 0
        else:
            min_index = self.number_of_preset_candidates
        return min_index
