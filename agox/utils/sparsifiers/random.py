import numpy as np

from agox.utils.sparsifiers.ABC_sparsifier import SparsifierBaseClass


class Random(SparsifierBaseClass):
    name = "Random"

    def sparsify(self, X: np.ndarray = None) -> np.ndarray:
        if self.m_points < X.shape[0]:
            m_indices = np.random.choice(X.shape[0], size=self.m_points, replace=False)
            Xm = X[m_indices, :]

        else:
            Xm = X

        return Xm, m_indices
