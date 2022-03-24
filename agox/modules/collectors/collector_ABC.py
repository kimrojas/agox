import numpy as np
from abc import ABC, abstractmethod

from agox.observer_handler import Observer

class CollectorBaseClass(ABC, Observer):

    def __init__(self, order=2, sets={'set_key':'candidates'}, gets={}):
        super().__init__(sets=sets, gets=gets, order=order)
        self.candidates = []

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def make_candidates(self):
        pass
    
    def generate_candidates(self):
        # Make the candidates - this is the method that differs between versions of the class. 
        self.make_candidates()

        # Add to the episode_cache:
        self.add_to_cache(self.set_key, self.candidates, 'a')

########################################################################################################################
# Methods for dealing with candidates
########################################################################################################################

    def get_current_candidates(self):
        """
        Return the current candidates.
        """
        return self.candidates

    def set_current_candidates(self, candidates, values=None):
        """
        Overwrite the current list of candidates, e.g. done by the acquisitor after having sorted the candidate objects.
        """

        if values is not None:
            assert len(candidates) == len(values)

        self.candidates = candidates
        if values is not None:
            self.values = values

    def pop_candidate(self):
        if self.candidates:
            return self.candidates.pop(0)
        else:
            return None

    def get_random_candidate(self):
        if len(self.candidates) == 0:
            return None
        index = np.random.randint(low=0, high=len(self.candidates))
        return self.candidates[index].copy()

########################################################################################################################
# Other methods
########################################################################################################################

    def assign_from_main(self, main):
        super().assign_from_main(main)
    
    def attach(self, main):
        main.attach_observer(self.name+'.generate_candidates', self.generate_candidates, order=self.order)   
    
    def plot_confinement(self):
        from agox.modules.helpers.plot_confinement import plot_confinement
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        for generator in self.generators: 
            if generator.confined:
                fig, ax = plot_confinement(self.environment.get_template(), generator.confinement_cell, generator.cell_corner)
                plt.savefig(f'confinement_plot_{generator.name}.png')
                plt.close()
