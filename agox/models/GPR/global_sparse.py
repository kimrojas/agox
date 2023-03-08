import numpy as np
from agox.models.GPR.ABC_sparse import SparseBaseClass


class GlobalSparseGPR(SparseBaseClass):

    name = 'GlobalSparseGPR'    

    def get_features(self, atoms):
        """
        Get features for a given atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        np.ndarray
            Features
        
        """
        return np.array(self.descriptor.get_global_features(atoms))

    
    def _make_L(self, atoms_list):
        """
        Make the L matrix

        Parameters
        ----------
        atoms_list : list of ase.Atoms
            List of ase.Atoms objects
        
        Returns
        -------
        np.ndarray
            L matrix
        
        """
        return np.eye(len(atoms_list))

    
    def _update_L(self, new_atoms_list):
        """
        Update the L matrix

        Parameters
        ----------
        new_atoms_list : list of ase.Atoms
            List of ase.Atoms objects

        Returns
        -------
        np.ndarray
            L matrix
        
        """
        new_size = len(new_atoms_list) + self.L.shape[0]
        return np.eye(new_size)
    
