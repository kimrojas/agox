from agox.modules.samplers.sampler_ABC import SamplerBaseClass
import numpy as np

class DummySampler(SamplerBaseClass):

    name = 'DumboTheDummySamplo'
    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_random_member(self):
        all_candidates = self.database.get_all_candidates()
        if len(all_candidates) == 0:
            return None
        else:
            idx = int(np.random.randint(low=0, high=len(all_candidates), size=1)[0])
            return all_candidates[idx].copy()

    def setup(self):
        pass

    def assign_from_main(self, main):
        super().assign_from_main(main)