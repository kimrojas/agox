from abc import ABC, abstractmethod
import numpy as np
from ase.io import write
from agox.observer_handler import Observer

class SamplerBaseClass(ABC, Observer):

    def __init__(self, sets={}, gets={}, order=1):
        super().__init__(sets=sets, gets={}, order=order)
        self.sample = []

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

    @abstractmethod
    def setup(self):
        pass

    ########################################################################################
    # Default methods
    ########################################################################################

    def get_random_member_with_calculator(self):
        return self.get_random_member(copy_calculator=True)

    def assign_from_main(self, main):
        super().assign_from_main(main)
    
    def attach(self, main):
        main.attach_observer(self.name+'.setup', self.setup, order=self.order)

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

    def get_template(self):
        """
        Some Generators may not use a fully-build structure as the 'parent' or 'Generator-seed', but 
        a template (and information about which atoms to place.)
        """
        #return self.template.copy()
        return self.environment.get_template() # ? 
    

