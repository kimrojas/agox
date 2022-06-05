from abc import ABC, abstractmethod
from agox.observer import Observer, ObserverHandler

from ase.calculators.singlepoint import SinglePointCalculator
from agox.modules.helpers.writer import header_footer, Writer

from ase import Atoms

class DatabaseBaseClass(ABC, ObserverHandler, Observer, Writer):

    def __init__(self, gets={'get_key':'evaluated_candidates'}, sets={}, order=6, verbose=True, use_counter=True, prefix=''):
        Observer.__init__(self, gets=gets, sets=sets, order=order)        
        ObserverHandler.__init__(self)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)
        self.candidates = []

        self.objects_to_assign = []

        self.add_observer_method(self.store_in_database, sets=self.sets[0], gets=self.gets[0], order=self.order[0])

    ########################################################################################
    # Required methods                                                          
    ########################################################################################

    @abstractmethod
    # def write(self, positions, energy, atom_numbers, cell, **kwargs):
    def write(self, grid):
        """
        Write stuff to database
        """

    def add_objects_to_assign_to(self, class_object):
        self.objects_to_assign.append(class_object)

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

    @header_footer
    def store_in_database(self):
        
        evaluated_candidates = self.get_from_cache(self.get_key)
        anything_accepted = False
        for j, candidate in enumerate(evaluated_candidates):

            # Dispatch to observers only when adding the last candidate. 
            dispatch = (j+1) == len(evaluated_candidates)

            if candidate: 
                self.writer('Energy {:06d}: {}'.format(self.get_iteration_counter(), candidate.get_potential_energy()), flush=True)
                self.store_candidate(candidate, accepted=True, write=True, dispatch=dispatch)
                anything_accepted = True

            elif candidate is None:
                dummy_candidate = self.candidate_instantiator(template=Atoms())
                dummy_candidate.set_calculator(SinglePointCalculator(dummy_candidate, energy=float('nan')))

                # This will dispatch to observers if valid data has been added but the last candidate is None. 
                self.store_candidate(candidate, accepted=False, write=True, dispatch=bool(anything_accepted*dispatch))

    def assign_from_main(self, main):
        super().assign_from_main(main)
        for observer_object in self.objects_to_assign:
            observer_object.assign_from_main(main)
