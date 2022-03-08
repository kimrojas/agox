import numpy as np
from agox.modules.models.sparsifiers.sparsifier_ABC import SparsifierBaseClass

class DublicateSparsifier(SparsifierBaseClass):
    name = 'DublicateSparsifier'

    def __init__(self, n_latest=100, dublication_factor=2, **kwargs):
        super().__init__(**kwargs)
        self.n_latest = 100
        self.dublication_factor = dublication_factor

    def _sparsify(self, list_of_candidates):
        if len(list_of_candidates) < self.n_latest:
            return list_of_candidates
        else:
            return list_of_candidates[-self.n_latest:]*self.dublication_factor
            
