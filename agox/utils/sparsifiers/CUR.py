from typing import List, Optional

import numpy as np
from ase import Atoms
from scipy.linalg import svd

from agox.utils.sparsifiers.ABC_sparsifier import SparsifierBaseClass


class CUR(SparsifierBaseClass):
    name = "CUR"

    def sparsify(
        self, atoms: Optional[List[Atoms]] = None, X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        X = self.preprocess(atoms, X)

        if X.shape[0] < self.m_points:
            m_indices = np.arange(0, X.shape[0])
            return X, m_indices

        U, _, _ = svd(X)
        score = np.sum(U[:, : self.m_points] ** 2, axis=1) / self.m_points
        sorter = np.argsort(score)[::-1]
        Xm = X[sorter, :][: self.m_points, :]

        return Xm
