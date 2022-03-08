import numpy as np
from agox.modules.models.sparsifiers.sparsifier_ABC import SparsifierBaseClass

class LatestSparsifier(SparsifierBaseClass):
    name = 'LatestSparsifier'

    def __init__(self, n_latest=100, **kwargs):
        super().__init__(**kwargs)
        self.n_latest = n_latest

    def _sparsify(self, list_of_candidates):
        if len(list_of_candidates) < self.n_latest:
            return list_of_candidates
        else:
            return list_of_candidates[-self.n_latest:]
            

class RemoveFirstSparsifier(SparsifierBaseClass):
    name = 'RemoveFirstSparsifier'

    def __init__(self, n_first=100, **kwargs):
        super().__init__(**kwargs)
        self.n_first = n_first

    def _sparsify(self, list_of_candidates):
        if len(list_of_candidates) < self.n_first:
            return list_of_candidates
        else:
            return list_of_candidates[self.n_first:]
