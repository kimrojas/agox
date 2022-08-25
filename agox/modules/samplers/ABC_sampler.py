from abc import ABC, abstractmethod
import numpy as np
from ase.io import write
from agox.modules.helpers.writer import header_footer, Writer
from agox.observer import Observer

class SamplerBaseClass(ABC, Observer, Writer):

    def __init__(self, sets={}, gets={}, order=1, verbose=True, use_counter=True, prefix=''):
        Observer.__init__(self, sets=sets, gets=gets, order=order)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)
        self.sample = []

        self.add_observer_method(self.setup_sampler, sets=self.sets[0], gets=self.gets[0], order=self.order[0])

    ########################################################################################
    # Required properties
    ########################################################################################    

    @property
    @abstractmethod
    def name(self):
        return NotImplementedError

    ########################################################################################
    # Required methods
    ########################################################################################
    
    @abstractmethod
    def get_random_member(self):
        pass
        
    def get_all_members(self):
        return [candidate.copy() for candidate in self.sample]

    @abstractmethod
    def setup(self):
        pass

    ########################################################################################
    # Default methods
    ########################################################################################

    def assign_from_main(self, main):
        super().assign_from_main(main)
    
    @header_footer
    def setup_sampler(self):
        self.setup()

    def get_random_member(self):
        if len(self.sample) == 0:
            return None
        index = np.random.randint(low=0, high=len(self.sample))
        member = self.sample[index].copy()
        member.add_meta_information('sample_index', index)
        return member

    def get_random_member_with_calculator(self):
        if len(self.sample) == 0:
            return None
        index = np.random.randint(low=0, high=len(self.sample))
        member = self.sample[index].copy()
        member.add_meta_information('sample_index', index)
        self.sample[index].copy_calculator_to(member)
        return member    

    def __len__(self):
        return len(self.sample)
