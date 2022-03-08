from ase import atoms
from agox.modules.generators.generator_ABC import GeneratorBaseClass
import numpy as np
from ase.data import covalent_radii
from ase import Atoms
from scipy.spatial.distance import cdist

class HemisphereGenerator(GeneratorBaseClass):

    name = 'HemisphereGenerator'

    def __init__(self, selection_percentages={'low':0.25, 'high':0.5}, max_number_of_attempts=100, extra_radius_amplitude=1, extra_radius_params={'low':-0.5, 'high':3}, **kwargs):
        super().__init__(**kwargs)
        self.selection_percentages = selection_percentages
        self.max_number_of_attempts = max_number_of_attempts
        self.extra_radius_amplitude = extra_radius_amplitude
        self.extra_radius_params = extra_radius_params

    def get_candidates(self, sampler, environment):

        # Get a candidate from the sampler:
        candidate = sampler.get_random_member()
        if candidate is None:
            return [None]

        cluster_size = len(candidate) - len(candidate.template)
        cluster_indices = np.arange(len(candidate.template), len(candidate))

        # Calculate the center of geometry of the: 
        cog = np.mean(candidate.positions[cluster_indices], axis=0)
        
        # Calculate the distance to the COG: 
        cog_distances = np.zeros(len(cluster_indices))
        for i, ci in enumerate(cluster_indices):
            cog_distances[i] = np.linalg.norm(candidate.positions[ci, :]-cog)

        # Pick the ones that are furthest away: 
        atoms_picked = np.floor(np.random.uniform(**self.selection_percentages) * cluster_size).astype(int)

        picked_cluster_indices = cluster_indices[np.flip(np.argsort(cog_distances))[0:atoms_picked]]
        numbers = candidate.get_atomic_numbers()[picked_cluster_indices]
        del candidate[picked_cluster_indices]
        radius = np.mean(cog_distances)
        num_attempts = 0
        for number in numbers:
            succesful_position_found = False
            while not succesful_position_found and num_attempts < self.max_number_of_attempts:
                current_radius = radius + self.extra_radius_amplitude * np.random.uniform(**self.extra_radius_params)
                suggested_position = self.sample_hemisphere(current_radius) + cog
                num_attempts += 1
                if self.check_new_position(candidate, suggested_position, number) and self.check_confinement(suggested_position):
                    candidate += Atoms(numbers=[number], positions=suggested_position.reshape(1, 3))
                    succesful_position_found = True

        if num_attempts >= self.max_number_of_attempts:
            return [None]

        return [candidate]

    def sample_hemisphere(self, radius):
        theta = np.random.uniform(0, np.pi/2)
        phi = np.random.uniform(0, 2*np.pi)
        return radius * np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
                        



        


