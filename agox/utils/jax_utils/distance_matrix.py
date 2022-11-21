from warnings import warn
import jax.numpy as np
from ase import Atoms

from agox.utils.jax_utils.distance.helpers import distance_matrix, periodicity

class Distance():
    def __init__(self, mic: bool=True, wrap: bool=False):
        self.mic = mic
        self.wrap = wrap

    def __call__(self, atoms: Atoms) -> np.ndarray:
        return self.calculate(atoms)

    def calculate(self, atoms: Atoms) -> np.ndarray:
        pbc = atoms.get_pbc()
        if self.wrap:
            atoms.set_pbc((True, True, True))
            atoms.wrap()
            atoms.set_pbc(pbc)

        if not self.mic:
            p = 'no'
        else:
            names = ['x', 'y', 'z']
            if sum(pbc) == 0:
                p = 'no'
            else:
                p = ''.join([names[i] for i in range(3) if pbc[i]])

        cell = atoms.get_cell().array
        if np.sum(cell-np.diag(np.diagonal(cell))) > 1e-4:
            warn('get_all_distance() call may not work as inteded for non-square cells.')

        return distance_matrix(atoms.get_positions(), cell, periodicity[p])

    
