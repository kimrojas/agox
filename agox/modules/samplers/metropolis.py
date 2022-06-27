from agox.modules.samplers.ABC_sampler import SamplerBaseClass
import numpy as np

class MetropolisSampler(SamplerBaseClass):

    name = 'MetropolisSampler'

    def __init__(self, database=None, temperature=1, **kwargs):
        """
        Temperature: The selection temperature in eV. 
        """
        super().__init__(**kwargs)

        self.temperature = temperature
        self.chosen_candidate = None
        assert database is not None
        self.database = database

    def get_random_member(self):
        if self.chosen_candidate is not None:
            return self.chosen_candidate.copy()
        else:
            return None

    def setup(self):
        potential_step = self.get_candidate_to_consider()
        if potential_step is None:
            return
        if self.chosen_candidate is None:
            self.chosen_candidate = potential_step
            if self.chosen_candidate is not None:
                self.chosen_candidate.add_meta_information('accepted', True)
        else:
            Eold = self.chosen_candidate.get_potential_energy()
            Enew = potential_step.get_potential_energy()
            if Enew < Eold:
                self.chosen_candidate = potential_step
                accepted = True
            else:
                P = np.exp(-(Enew - Eold)/self.temperature)
                r = np.random.rand()
                if r < P:
                    accepted = True
                    self.chosen_candidate = potential_step
                else:
                    accepted = False
            if self.verbose:
                if Enew < Eold:
                    P = 1
                    r = 0
                self.writer(f'METROPOLIS: Eold:{Eold:8.3f} Enew:{Enew:8.3f} T:{self.temperature:8.3f} P:{P:8.3f} r:{r:8.3f}')
            potential_step.add_meta_information('accepted', accepted)

        return potential_step

        # Other things may rely on the length of the sample.
        self.sample = [self.chosen_candidate]

    def assign_from_main(self, main):
        super().assign_from_main(main)


    def get_candidate_to_consider(self):        
        candidates = self.get_from_cache(self.get_key)

        if candidates is None or not len(candidates) > 0:
            return None
        return candidates[-1]   # take the latest evaluated candidate

