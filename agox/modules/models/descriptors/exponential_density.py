import numpy as np
from agox.modules.models.descriptors.descriptor_ABC import DescriptorBaseClass

from scipy.spatial.distance import cdist

class ExponentialDensity(DescriptorBaseClass):

    def __init__(self, atomic_numbers, sigma=[1]):
        self.atomic_numbers = atomic_numbers
        self.sigma = np.array(sigma)

        self.n_features = len(self.atomic_numbers) * len(self.sigma)

    def get_feature(self, atoms):

        # Get distance matrix:
        distances = cdist(atoms.positions, atoms.positions)

        numbers = atoms.get_atomic_numbers()

        F = np.zeros((len(atoms), self.n_features))

        where_dict = {atomic_number:np.argwhere(numbers==atomic_number) for atomic_number in self.atomic_numbers}

        for i in range(len(atoms)):
            for c, atomic_number in enumerate(self.atomic_numbers):
                indices = where_dict[atomic_number]
                F[i, c*len(self.sigma):(c+1)*len(self.sigma)] = np.sum(np.exp(-self.sigma * distances[i, indices]), axis=0)
        return F

    
        