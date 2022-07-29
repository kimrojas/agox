import numpy as np
from agox.modules.models.descriptors.descriptor_ABC import DescriptorBaseClass

from scipy.spatial.distance import cdist

class ExponentialDensity(DescriptorBaseClass):

    def __init__(self, atomic_numbers, sigma=[1], rc = 11.9):
        self.atomic_numbers = atomic_numbers
        self.sigma = np.array(sigma)
        self.rc = rc

        self.n_features = len(self.atomic_numbers) * len(self.sigma)

    def get_feature(self, atoms):

        # Get distance matrix:
        distances = cdist(atoms.positions, atoms.positions)

        numbers = atoms.get_atomic_numbers()

        F = np.zeros((len(atoms), self.n_features + 1))

        where_dict = {atomic_number:np.argwhere(numbers==atomic_number) for atomic_number in self.atomic_numbers}

        for i in range(len(atoms)):
            for c, atomic_number in enumerate(self.atomic_numbers):
                indices = where_dict[atomic_number]
                F[i, c*len(self.sigma):(c+1)*len(self.sigma)] = np.sum(np.exp(- distances[i, indices] / self.sigma) * self.gc(distances[i, indices]) / self.sigma, axis=0)
        
        F[:, -1] = atoms.get_atomic_numbers()
        return F

    def gc(self, distances):
#        print(distances)
        filter = np.logical_and(distances <= self.rc, distances > 0)
#        filter = filter.reshape(-1)
#        print(filter)
#        print(filter.shape)
#        print(filter.reshape(-1).shape)
#        filter = filter.reshape(-1)
        values = np.zeros((len(distances), 1))
#        print(values)
#        print(values.shape)
#        print(distances[filter])
#        print(values[filter])
#        print(0.5 * np.cos(np.pi * distances[filter] / self.rc) + 0.5)
        values[filter] += 0.5 * np.cos(np.pi * distances[filter] / self.rc) + 0.5
        return values
    
        