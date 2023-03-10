import numpy as np
from agox.models.GPR.sparsifiers.ABC_sparsifier import SparsifierBaseClass

class Random(SparsifierBaseClass):
    
    name = 'Random'
    
    def sparsify(self, X):
        if self.m_points > self.Xn.shape[0]:
            m_indices = np.arange(0,self.Xn.shape[0])
        else:
            m_indices = np.random.choice(self.Xn.shape[0], size=self.m_points, replace=False)
        Xm = self.Xn[m_indices, :]
        
        return Xm, m_indices



