import numpy as np
from agox.modules.models.descriptors.descriptor_ABC import DescriptorBaseClass

class CartesianCoordinate(DescriptorBaseClass):
    def __init__(self, dim = 1, append_atomic_number = False):
        self.dim = 1
        self.append_atomic_number = append_atomic_number

    def get_feature(self, atoms):
        if self.append_atomic_number:
            F = np.zeros((len(atoms), self.dim + 1))
        else:
            F = np.zeros((len(atoms), self.dim))

        positions = atoms.get_positions()
        F[:, :self.dim] = positions[:, :self.dim]
        if self.append_atomic_number:
            F[:, -1] = atoms.get_atomic_numbers()
        return F