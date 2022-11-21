from warnings import warn
import jax.numpy as np
from ase import Atoms
from typing import Tuple, Optional

from agox.utils.jax_utils.distance.helpers import all_distance_vectors, get_repeats, periodicity, diff, diff_vectorized

## Unresolved issues with speed:
# - how to calculate repeats faster
# - how to calculate argwhere faster

class Neighborlist():
    def __init__(self, rcut: float=4., wrap: bool=False):
        self.rcut = rcut
        self.wrap = wrap

    def __call__(self, atoms: Atoms) -> None:
        update(atoms)

    def update(self, atoms: Atoms) -> None:
        pbc = atoms.get_pbc()
        if self.wrap:
            atoms.set_pbc((True, True, True))
            atoms.wrap()
            atoms.set_pbc(pbc)

        self.positions = atoms.get_positions()
        self.cell = atoms.get_cell().array
        self.pbc = np.array(atoms.pbc).astype(float)
        self.repeats = get_repeats(self.cell, self.rcut, self.pbc)
        #self.repeats = periodicity['xyz']

        self.all_dists, self.cells = all_distance_vectors(self.positions,
                                                          self.cell, self.repeats)

        
    def get_neighbors(self, index:int,
                      indices:Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Contrary to ASE neighborlist this returns the offsets @ cell, hence the vector
        that translates the atom to the correct location. 
        """
        if indices is None:
            indices = np.argwhere(self.all_dists[index,:,:]<self.rcut)
            
        return indices[:,0], self.cells[indices[:,1]]

    def get_neighbors_distances(self, index: int,
                                indices:Optional[np.ndarray]=None) -> np.ndarray:
        if indices is None:
            indices = np.argwhere(self.all_dists[index,:,:] < self.rcut)
        return self.all_dists[index, indices[:,0], indices[:, 1]]

    def get_neighbors_indices(self, index:int) -> np.ndarray:
        return np.argwhere(self.all_dists[index,:,:]<self.rcut)
    

    def get_gradient(self, i: int, j:int, repeat_index) -> np.ndarray:
        """
        Returns gradient for distance r_ij with repect to r_i. 
        """
        print(self.positions[j,:] + self.repeats[repeat_index, :])
        return -diff(self.positions[i,:], self.positions[j,:] + self.repeats[repeat_index, :])/self.all_dists[i,j,repeat_index]


    def get_gradients(self, i: int, indices: np.ndarray) -> np.ndarray:
        """
        Returns gradient for distance r_ij with repect to r_i for all neighbors j. 
        """
        r = self.positions[indices[:,0], :] + self.repeats[indices[:,1], :]
        rij = self.all_dists[i, indices[:,0], indices[:, 1]].reshape(-1, 1)
        return -diff_vectorized(self.positions[i,:], r)/rij
