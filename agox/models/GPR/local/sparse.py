import numpy as np
from agox.models.GPR.ABC_sparse import SparseBaseClass


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
        return np.vstack(self.descriptor.get_local_features(atoms))


