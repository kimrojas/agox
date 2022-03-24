from abc import ABC, abstractmethod
from agox.observer_handler import Observer, ObserverHandler

from ase.calculators.singlepoint import SinglePointCalculator

class DatabaseBaseClass(ABC, ObserverHandler, Observer):

    def __init__(self, gets={'get_key':'evaluated_candidates'}, sets={}, order=6):
        Observer.__init__(self, gets=gets, sets=sets, order=order)
        ObserverHandler.__init__(self)
        self.candidates = []

    ########################################################################################
    # Required methods                                                          
    ########################################################################################

    @abstractmethod
    # def write(self, positions, energy, atom_numbers, cell, **kwargs):
    def write(self, grid):
        """
        Write stuff to database
        """

    @abstractmethod
    def store_candidate(self,candidate_object):
        pass

    @abstractmethod
    def get_all_candidates(self):
        pass

    @property
    @abstractmethod
    def name(self):
        return NotImplementedError
    
    ########################################################################################
    # Default methods
    ########################################################################################

    def __len__(self):
        return len(self.candidates) 

    def store_in_database(self):
        
        gauged_candidates = self.get_from_cache(self.get_key)
        anything_accepted = False
        for j, candidate in enumerate(gauged_candidates):

            # Dispatch to observers only when adding the last candidate. 
            dispatch = (j+1) == len(gauged_candidates)

            if candidate: 
                print('Energy {:06d}: {}'.format(self.get_episode_counter(), candidate.get_potential_energy()), flush=True)
                self.store_candidate(candidate, accepted=True, write=True, dispatch=dispatch)
                anything_accepted = True

            elif candidate is None:
                dummy_candidate = self.candidate_instantiator(template=Atoms())
                dummy_candidate.set_calculator(SinglePointCalculator(dummy_candidate, energy=float('nan')))

                # This will dispatch to observers if valid data has been added but the last candidate is None. 
                self.store_candidate(candidate, accepted=False, write=True, dispatch=bool(anything_accepted*dispatch))

    def assign_from_main(self, main):
        super().assign_from_main(main)
    
    def attach(self, main):
        main.attach_observer(self.name+'.store_in_database', self.store_in_database, order=self.order)
