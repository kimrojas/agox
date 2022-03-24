from agox.modules.samplers.sampler_ABC import SamplerBaseClass
import numpy as np

class MetropolisSampler(SamplerBaseClass):

    name = 'MetropolisSampler'

    def __init__(self, database=None, temperature=1):
        """
        Temperature: The selection temperature in eV. 
        """
        super().__init__()

        self.temperature = temperature
        self.chosen_candidate = None
        assert database is not None
        self.database = database

    def get_random_member(self):
        return self.chosen_candidate.copy()

    def setup(self):
        if self.chosen_candidate is None:
            self.chosen_candidate = self.database.get_most_recent_candidate()
        else:
            potential_step = self.database.get_most_recent_candidate()

            Eold = self.chosen_candidate.get_potential_energy()
            Enew = potential_step.get_potential_energy()
            if Enew < Eold:
                self.chosen_candidate = potential_step
            else:
                P = np.exp(-(Enew - Eold)/self.temperature)
                r = np.random.rand()
                if r < P:
                    self.chosen_candidate = potential_step

    def assign_from_main(self, main):
        super().assign_from_main(main)