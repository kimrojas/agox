import numpy as np
from copy import copy

from agox.modules.models.descriptors.soap import SOAP

from ase.data import covalent_radii
def get_relative_distances(atoms):
    distances_abs = atoms.get_all_distances(mic=True)

    numbers = atoms.get_atomic_numbers()
    r = [covalent_radii[number] for number in numbers]
    x,y = np.meshgrid(r,r)
    optimal_distances = x+y

    distances_rel = distances_abs / optimal_distances

    return distances_rel

def shortest_relative_bond(atoms):
    d = get_relative_distances(atoms)
    d += np.eye(d.shape[0])
    return np.min(d,axis=1)

class ExtendedSOAP(SOAP):

    def __init__(self, *args, extension_strength=50, fermi_dirac_center=0.6, fermi_dirac_width=0.1, **kwargs):
        self.extension_strength = extension_strength
        self.fermi_dirac_center = fermi_dirac_center
        self.fermi_dirac_width = fermi_dirac_width
        super().__init__(*args, **kwargs)

    def _soap_create(self, atoms):

        f = lambda x: 1/(1 + np.exp(-(x-self.fermi_dirac_center)/self.fermi_dirac_width))

        advanced_descriptor = self.soap.create(atoms)
        relative_bond = shortest_relative_bond(atoms)

        advanced_descriptor = (advanced_descriptor.T * f(relative_bond)).T
        extension = self.extension_strength * (1 - f(relative_bond))
        extension = extension.reshape(-1,1)
        extended_descriptor = np.hstack((advanced_descriptor,extension))
        return extended_descriptor

    def get_feature(self, atoms):
        """Returns soap descriptor for "atoms".
        Dimension of output is [n_centers, n_features]
        """
        assert False, 'not implemented'    

    def get_feature_derivatives(self, atoms):
        """Returns derivative of soap descriptor for "atoms" with
        respect to atomic coordinates.
        Dimension of output is [n_centers, 3*n_atoms, n_features]
        """
        assert False, 'not implemented'

    def get_local_environments(self, atoms):
        """Returns local environments for all "atoms".
        Dimensions of output (num local environents) x (lenght descriptor) as array
        """
        if isinstance(atoms, list):
            features = np.vstack([self._soap_create(atom) for atom in atoms])
        else:
            features = self._soap_create(atoms)
        if self.normalize:
            ddot = np.dot(features, features.T)
            return features/np.sqrt(np.diagonal(ddot)[:,np.newaxis])
        else:
            return features

