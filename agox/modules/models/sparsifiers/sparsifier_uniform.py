import numpy as np
from agox.modules.models.sparsifiers.sparsifier_ABC import SparsifierBaseClass

class UniformSparsifier(SparsifierBaseClass):
    name = 'UniformSparsifier'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _sparsify(self, list_of_candidates):
        if len(list_of_candidates) < self.n_max:
            return list_of_candidates
        else:
            idxs = np.random.randint(0,len(list_of_candidates), self.n_max)
            return [list_of_candidates[i] for i in idxs]
            
