import numpy as np
from agox.modules.generators.ABC_generator import GeneratorBaseClass

class ComplementaryEnergyGenerator(GeneratorBaseClass):

    def __init__(self, complementary_calculator):
        self.ce_calculator = complementary_calculator

    def get_candidate(self, environment, sampler):
        
        # Get candidate from Sampler:
        candidate = sampler.get_random_candidate()
        if candidate is None:
            return [None]


from ase.calculators.calculator import Calculator
from scipy.spatial.distance import cdist

class ComplementaryEnergyCalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(self, descriptor, attractors, sigma=10, dx=0.01):
        super().__init__()
        self.descriptor = descriptor
        self.attractors = attractors
        self.sigma = sigma
        self.dx = dx

    def get_ce_energy(self, atoms):
        feature = self.descriptor.get_feature(atoms)
        CE = 0
        for a in range(len(atoms)):            
            # Determine attractor:
            attractor_index = np.argmin(cdist(feature[a].reshape(1, -1), self.attractors))
            attractor = self.attractors[attractor_index]
            CE += -np.exp(-np.linalg.norm(feature[a]-attractor)**2/(2*self.sigma**2))

        return CE

    def calculate(self, atoms, properties=[], *args, **kwargs):
        #super().__init__(atoms, *args, **kwargs)
        self.results['energy'] = self.get_ce_energy(atoms)
        if 'forces' in properties:
            self.results['forces'] = self.get_numerical_forces(atoms)
    
    def get_numerical_forces(self, atoms):
        F = np.zeros((len(atoms), 3))
        for a in range(len(atoms)):
            for d in range(3):                
                atoms.positions[a, d] += self.dx
                ef = self.get_ce_energy(atoms)
                atoms.positions[a, d] -= 2 * self.dx
                em = self.get_ce_energy(atoms)
                atoms.positions[a, d] += self.dx

                F[a, d] = -(ef-em)/(2*self.dx)
        return F

class ComplementaryEnergyDistanceCalculator(ComplementaryEnergyCalculator):

    def __init__(self, descriptor, attractors, sigma=10):
        super().__init__(descriptor, attractors)
        self.descriptor = descriptor
        self.attractors = attractors
        self.sigma = sigma


    def get_ce_energy(self, atoms):
        feature = self.descriptor.get_feature(atoms)
        CE = 0
        for a in range(len(atoms)):            
            # Determine attractor:
            attractor_index = np.argmin(cdist(feature[a].reshape(1, -1), self.attractors))
            attractor = self.attractors[attractor_index]
            CE += np.linalg.norm(feature[a]-attractor)

        return CE

class ComplementaryEnergyDistanceSquaredCalculator(ComplementaryEnergyCalculator):

    def __init__(self, descriptor, attractors, sigma=10):
        super().__init__(descriptor, attractors)
        self.descriptor = descriptor
        self.attractors = attractors
        self.sigma = sigma


    def get_ce_energy(self, atoms):
        feature = self.descriptor.get_feature(atoms)
        CE = 0
        for a in range(len(atoms)):            
            # Determine attractor:
            attractor_index = np.argmin(cdist(feature[a].reshape(1, -1), self.attractors))
            attractor = self.attractors[attractor_index]
            CE += np.linalg.norm(feature[a]-attractor) ** 2

        return CE



        

    