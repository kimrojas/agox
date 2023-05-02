import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator
from scipy.spatial.distance import cdist
from matscipy.neighbours import neighbour_list as msp_nl

class ComplementaryEnergyGaussianCalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(self, descriptor, attractors = None, mover_indices = None, sigma=10, dx=0.001):
        super().__init__()
        self.descriptor = descriptor
        self.attractors = attractors
        self.mover_indices = mover_indices
        self.sigma = sigma
        self.dx = dx

    def calculate(self, atoms, properties=[], *args, **kwargs):
        self.results['energy'] = self.get_ce_energy(atoms)
        if 'forces' in properties:
            self.results['forces'] = self.get_numerical_forces(atoms)
            
    def get_local_ce_energy(self, atoms):
        features = self.descriptor.get_local_features(atoms)
        CE = []
        for a in self.mover_indices:
            attractor_index = np.argmin(cdist(features[a].reshape(1, -1), self.attractors))
            attractor = self.attractors[attractor_index]
            CE.append(np.exp(-np.linalg.norm(features[a]-attractor)**2/(2*self.sigma**2)))

        CE = np.array(CE)
        return CE

    def get_ce_energy(self, atoms):
        features = self.descriptor.get_local_features(atoms)
        CE = 0
        for a in self.mover_indices:            
            attractor_index = np.argmin(cdist(features[a].reshape(1, -1), self.attractors))
            attractor = self.attractors[attractor_index]
            CE += -np.exp(-np.linalg.norm(features[a]-attractor)**2/(2*self.sigma**2))

        return CE
    
    def get_numerical_forces(self, atoms):
        F = np.zeros((len(atoms), 3))
        for a in self.mover_indices:
            for d in range(3):                
                atoms.positions[a, d] += self.dx
                ef = self.get_ce_energy(atoms)
                atoms.positions[a, d] -= 2 * self.dx
                em = self.get_ce_energy(atoms)
                atoms.positions[a, d] += self.dx

                F[a, d] = -(ef-em)/(2*self.dx)
        return F

class ComplementaryEnergyKernelMethod(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(self, descriptor, attractors = None, mover_indices = None, sigma = 10, dx=0.001):
        super().__init__()
        self.descriptor = descriptor
        self.attractors = attractors
        self.mover_indices = mover_indices
        self.sigma = sigma
        self.dx = dx

    def calculate(self, atoms, properties=[], *args, **kwargs):
        self.results['energy'] = self.get_ce_energy(atoms)
        if 'forces' in properties:
            self.results['forces'] = self.get_numerical_forces(atoms)

    def get_local_ce_energy(self, atoms):
        features = self.descriptor.get_local_features(atoms)

        CE = []
        for a in self.mover_indices:
            e = -np.exp(-cdist(features[a].reshape(1, -1), self.attractors)**2/(2*self.sigma**2))
            CE.append(np.sum(e))

        CE = np.array(CE)
        return CE

    def get_ce_energy(self, atoms):
        features = self.descriptor.get_local_features(atoms)

        CE = 0
        for a in self.mover_indices:           
            e = -np.exp(-cdist(features[a].reshape(1, -1), self.attractors)**2/(2*self.sigma**2))
            CE += np.sum(e)

        return CE

    def get_numerical_forces(self, atoms):
        F = np.zeros((len(atoms), 3))
        for a in self.mover_indices:
            for d in range(3):                
                atoms.positions[a, d] += self.dx
                ef = self.get_ce_energy(atoms)
                atoms.positions[a, d] -= 2 * self.dx
                em = self.get_ce_energy(atoms)
                atoms.positions[a, d] += self.dx

                F[a, d] = -(ef-em)/(2*self.dx)
        return F

class ComplementaryEnergyDistanceCalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(self, descriptor, attractors = None, mover_indices = None, exponent = 1, dx=0.001):
        super().__init__()
        self.exponent = exponent
        self.descriptor = descriptor
        self.attractors = attractors
        self.mover_indices = mover_indices
        self.dx = dx

    def calculate(self, atoms, properties=[], *args, **kwargs):
        self.results['energy'] = self.get_ce_energy(atoms)
        if 'forces' in properties:
#            self.results['forces'] = self.get_numerical_forces(atoms)
            self.results['forces'] = self.get_forces(atoms)

    def get_local_ce_energy(self, atoms):
        features = self.descriptor.get_local_features(atoms)
        CE = []
        for a in self.mover_indices:
            attractor_index = np.argmin(cdist(features[a].reshape(1, -1), self.attractors))
            attractor = self.attractors[attractor_index]
            CE.append(np.linalg.norm(features[a]-attractor) ** self.exponent)

        CE = np.array(CE)
        return CE

    def get_ce_energy(self, atoms):
        features = self.descriptor.get_local_features(atoms)
        CE = 0
        for a in self.mover_indices:            
            attractor_index = np.argmin(cdist(features[a].reshape(1, -1), self.attractors))
            attractor = self.attractors[attractor_index]
            CE += np.linalg.norm(features[a]-attractor) ** self.exponent

        return CE

    def get_numerical_forces(self, atoms):
        F = np.zeros((len(atoms), 3))
        for a in self.mover_indices:
            for d in range(3):                
                atoms.positions[a, d] += self.dx
                ef = self.get_ce_energy(atoms)
                atoms.positions[a, d] -= 2 * self.dx
                em = self.get_ce_energy(atoms)
                atoms.positions[a, d] += self.dx

                F[a, d] = -(ef-em)/(2*self.dx)

        return F

    def get_forces(self, atoms):
        F = np.zeros((len(atoms), 3))

        features = self.descriptor.get_local_features(atoms)
        features_derivatives = self.descriptor.get_local_feature_gradient(atoms)
        mover_attractor_index = np.argmin(cdist(features, self.attractors), axis = 1)
        feature_dvec = features - self.attractors[mover_attractor_index]
        feature_dists = np.linalg.norm(feature_dvec, axis = 1)

        for i in self.mover_indices:
            if feature_dists[i] == 0:
                continue

            scalar_vector = self.exponent * feature_dvec[i] * feature_dists[i] ** (self.exponent - 2)
            scalar_vector = scalar_vector.reshape(-1) * features_derivatives[i]
            F[i, :] += np.sum(scalar_vector[i], axis = 1).reshape(-1)

            for j in self.mover_indices:
                if j != i:
                    F[j, :] += np.sum(scalar_vector[j], axis = 1).reshape(-1)

        return - F
