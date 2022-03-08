import numpy as np
from agox.modules.models.sparsifiers.sparsifier_ABC import SparsifierBaseClass

class NotLastSparsifier(SparsifierBaseClass):
    name = 'LatestSparsifier'

    def __init__(self, not_last=0.2, **kwargs):
        super().__init__(**kwargs)
        self.not_last = not_last

    def _sparsify(self, list_of_candidates):
        first_candidate_idx = int(np.rint(len(list_of_candidates)*self.not_last))
        if first_candidate_idx > 0:
            return list_of_candidates[first_candidate_idx:]
        else:
            return list_of_candidates
            
