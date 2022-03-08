import numpy as np
from agox.modules.models.sparsifiers.sparsifier_ABC import SparsifierBaseClass
from numpy.random import default_rng

class BestSparsifier(SparsifierBaseClass):
    name = 'BestSparsifier'

    def __init__(self, n_best=100, **kwargs):
        super().__init__(**kwargs)
        self.n_best = 100
        self.n_random = max([self.n_max-self.n_best, 0])
        self.rng = default_rng()

    def _sparsify(self, list_of_candidates):
        if len(list_of_candidates) < self.n_max:
            return list_of_candidates
        else:
            list_of_candidates.sort(key=lambda s: s.get_potential_energy())
            idxs = self.rng.choice(np.arange(self.n_best, len(list_of_candidates)), size=self.n_random, replace=False)
            return list_of_candidates[:self.n_best] + [list_of_candidates[i] for i in idxs]
            
