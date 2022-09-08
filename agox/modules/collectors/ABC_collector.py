import numpy as np
from abc import ABC, abstractmethod
from agox import observer
from agox.observer import Observer
from agox.modules.helpers.writer import header_footer, Writer

class CollectorBaseClass(ABC, Observer, Writer):

    def __init__(self, generators, sampler, environment, order=2, sets={'set_key':'candidates'}, gets={}, verbose=True, use_counter=True, prefix=''):
        Observer.__init__(self, sets=sets, gets=gets, order=order)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)

        self.generators = generators
        self.sampler = sampler
        self.environment = environment
        self.candidates = []
        self.plot_confinement()

        self.add_observer_method(self.generate_candidates, sets=self.sets[0], gets=self.gets[0], order=self.order[0])

        for generator in self.generators:            
            observer_methods = [observer_method for observer_method in generator.observer_methods.values()]
            for observer_method in observer_methods:
                generator.remove_observer_method(observer_method)

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def make_candidates(self):
        pass
    
    @abstractmethod
    def get_number_of_candidates(self):
        # Number of candidates to generate.
        pass

    @header_footer
    def generate_candidates(self):
        # Make the candidates - this is the method that differs between versions of the class. 
        self.make_candidates()

        # Add to the iteration_cache:
        self.add_to_cache(self.set_key, self.candidates, 'a')

########################################################################################################################
# Methods for dealing with candidates
########################################################################################################################

    def get_current_candidates(self):
        """
        Return the current candidates.
        """
        return self.candidates
        
########################################################################################################################
# Other methods
########################################################################################################################

    def assign_from_main(self, main):
        super().assign_from_main(main)
    
    def attach(self, main):
        super().attach(main)
        for generator in self.generators:
            generator.attach(main)

    def plot_confinement(self):
        for generator in self.generators: 
            generator.plot_confinement(self.environment)