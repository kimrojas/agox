import numpy as np
from agox.modules.models.sparsifiers.sparsifier_ABC import SparsifierBaseClass

class LatestBestSparsifier(SparsifierBaseClass):
    name = 'LatestBestSparsifier'

    def __init__(self, n_latest=100, n_best=50, **kwargs):
        super().__init__(**kwargs)
        self.n_latest = n_latest
        self.n_best = n_best

    def _sparsify(self, list_of_candidates):
        if len(list_of_candidates) < self.n_latest:
            return list_of_candidates
        else:
            energies = [c.get_potential_energy() for c in list_of_candidates]
            min_idxs = np.argsort(energies)[:self.n_best]
            best_candidates = [list_of_candidates[i] for i in min_idxs]
            return list_of_candidates[-self.n_latest:] + best_candidates
            
