import numpy as np
from agox.models.GPR.ABC_standard import StandardBaseClass


class GlobalGPR(StandardBaseClass):

    name = 'GlobalGPR'

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
