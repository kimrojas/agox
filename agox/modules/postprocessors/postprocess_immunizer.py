import numpy as np
from agox.modules.postprocessors.postprocess_ABC import PostprocessBaseClass

class Immunizer(PostprocessBaseClass):

    name = 'TheImmunizer'

    def __init__(self, probability=1):
        super().__init__()
        self.probability = probability

    def postprocess(self, candidate):
        if np.random.rand() < self.probability:
            candidate.set_postprocess_immunity(True)
        return candidate
        
class Deimmunizer(PostprocessBaseClass):

    name = 'TheDeimmunizer'

    def postprocess(self, candidate):        
        candidate.set_postprocess_immunity(False)
        return candidate
        