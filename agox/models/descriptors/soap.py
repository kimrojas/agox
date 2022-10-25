from copy import copy
import numpy as np
from dscribe.descriptors import SOAP as dscribeSOAP

from agox.models.descriptors.descriptor_ABC import DescriptorBaseClass





class SOAP(DescriptorBaseClass):

    def __init__(self, species, r_cut=4, nmax=3, lmax=2, sigma=1.0,
                 weight=True, periodic=True, dtype='float64', normalize=False, crossover=True):
        self.normalize = normalize
        
        if weight is True:
            weighting = {'function':'poly', 'r0':r_cut, 'm':2, 'c':1}
        elif weight is None:
            weighting = None
        elif weight is False:
            weighting = None
        else:
            weighting = weight
            

        self.soap = dscribeSOAP(
            species=species,
            periodic=periodic,
            rcut=r_cut,
            nmax=nmax,
            lmax=lmax,
            sigma=sigma,
            weighting=weighting,
            dtype=dtype,
            crossover=crossover,
            sparse=False)

        self.lenght = self.soap.get_number_of_features()
        print('SOAP lenght:', self.lenght)

    def get_feature(self, atoms):
        """Returns soap descriptor for "atoms".
        Dimension of output is [n_centers, n_features]
        """
        return self.soap.create(atoms)
            

    def get_feature_derivatives(self, atoms):
        """Returns derivative of soap descriptor for "atoms" with
        respect to atomic coordinates.
        Dimension of output is [n_centers, 3*n_atoms, n_features]
        """
        f_deriv = self.soap.derivatives(atoms, return_descriptor=False)
        n_centers, n_atoms, n_dim, n_features = f_deriv.shape
        return f_deriv.reshape(n_centers, n_dim*n_atoms, n_features)

    def get_local_environments(self, atoms):
        """Returns local environments for all "atoms".
        Dimensions of output (num local environents) x (lenght descriptor) as array
        """
        if isinstance(atoms, list):
            features = np.vstack([self.soap.create(atom) for atom in atoms])
        else:
            features = self.soap.create(atoms)
        if self.normalize:
            ddot = np.dot(features, features.T)
            return features/np.sqrt(np.diagonal(ddot)[:,np.newaxis])
        else:
            return features

