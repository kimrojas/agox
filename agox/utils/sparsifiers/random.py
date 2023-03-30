from typing import Optional, List

import numpy as np
from ase import Atoms

from agox.utils.sparsifiers.ABC_sparsifier import SparsifierBaseClass


class Random(SparsifierBaseClass):
    name = "Random"

    def sparsify(
        self, atoms: Optional[List[Atoms]] = None, X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        X = self.preprocess(atoms, X)
        
        if self.m_points < X.shape[0]:
            m_indices = np.random.choice(
                X.shape[0], size=self.m_points, replace=False
            )
            Xm = X[m_indices, :]

        else:
            Xm = X

        return Xm, m_indices
