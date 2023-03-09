import numpy as np
from agox.models.GPR.ABC_sGPR import SparseBaseClass


class LocalSparseGPR(SparseBaseClass):

    name = 'LocalSparseGPR'

    def get_features(self, atoms):
        """
        Get the features for a given ase.Atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            ase.Atoms object

        Returns
        -------
        np.ndarray
            Features for the ase.Atoms object
        
        """
        f = self.descriptor.get_local_features(atoms)

        if isinstance(f, np.ndarray) and len(f.shape) == 1:
            f = f.reshape(1, -1)
        f = np.vstack(f)
        return f
        # try:
        #     print(self.descriptor.get_local_features(atoms).shape)
        # except:
        #     print('first element:', self.descriptor.get_local_features(atoms)[0].shape)
            
        # return np.vstack(self.descriptor.get_local_features(atoms))


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
        lengths = [len(atoms) for atoms in atoms_list]
        r = len(lengths); c = np.sum(lengths)
        
        col = 0
        L = np.zeros((r,c))
        for i, atoms in enumerate(atoms_list):
            L[i,col:col+len(atoms)] = 1.
            col += len(atoms)
        return L

    
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
        new_lengths = [len(atoms) for atoms in new_atoms_list]
        size = len(new_lengths)
        new_total_length = np.sum(new_lengths)
        new_L = np.zeros((self.L.shape[0]+size, self.L.shape[1]+new_total_length))
        new_L[0:self.L.shape[0], 0:self.L.shape[1]] = self.L

        for l in range(size):
            step = int(np.sum(new_lengths[:l]))
            new_L[l+self.L.shape[0], (self.L.shape[1]+step):(self.L.shape[1]+step+new_lengths[l])] = 1            
        return new_L
    
