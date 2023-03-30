from typing import List, Tuple, Optional

import numpy as np
from ase import Atoms
from scipy.linalg import svd

from agox.utils.selectors.ABC_selector import SelectorBaseClass


class CUR(SelectorBaseClass):
    name = "CUR"

    def __init__(self, m_points: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.m_points = m_points

    def select(
        self,
        indices: Optional[np.ndarray] = None,
        atoms: Optional[List[Atoms]] = None,
        X: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[Optional[np.ndarray], Optional[List[Atoms]], Optional[np.ndarray]]:
        

        if X.shape[0] < self.m_points:
            return indices, atoms, X

        U, _, _ = svd(X)
        score = np.sum(U[:, : self.m_points] ** 2, axis=1) / self.m_points
        sorter = np.argsort(score)[::-1]
        Xm = X[sorter, :][: self.m_points, :]

        m_indices = sorter[: self.m_points]

        if 'atoms' in self.output:
            selected_atoms = atoms[m_indices]
        else:
            selected_atoms = None
            
        if indices in self.output:
            selected_indices = m_indices
        else:
            selected_indices = None

        return selected_indices, selected_atoms, Xm




    @property
    def implemented_output(self):
        return ['indices', 'features', 'atoms']

    @property
    def required_input(self):
        return ['features']
    
    @property
    def implemented_input(self):
        return ['indices', 'features', 'atoms']


