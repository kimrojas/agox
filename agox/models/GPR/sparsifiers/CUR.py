import numpy as np
from agox.models.GPR.sparsifiers.ABCSparsifier import SparsifierBaseClass
from scipy.linalg import svd

class CUR(SparsifierBaseClass):
    
    def sparsify(self, X):
        if self.Xn.shape[0] < self.m_points:
            m_indices = np.arange(0,self.Xn.shape[0])
            return X, m_indices

        U, _, _ = svd(X)
        score = np.sum(U[:,:self.m_points]**2, axis=1)/self.m_points
        sorter = np.argsort(score)[::-1]
        X = self.X[sorter, :][:self.m_points, :]

        return X, sorter[:self.m_points]

        
