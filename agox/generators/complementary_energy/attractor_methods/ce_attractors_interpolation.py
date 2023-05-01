import numpy as np
from agox.generators.complementary_energy.attractor_methods.ABC_attractors import AttractorMethodBaseClass

class AttractorInterpolation(AttractorMethodBaseClass):

    name = 'AttractorInterpolation'

    def __init__(self, descriptor, possible_number_of_attractors = [3, 6], attractors_from_template = True, predefined_attractors = None, order = 5):
        super().__init__(order = order)
        self.descriptor = descriptor
        self.possible_number_of_attractors = possible_number_of_attractors
        self.attractors_from_template = attractors_from_template
        self.predefined_attractors = predefined_attractors

    def get_attractors(self, structure):

        if self.predefined_attractors is not None:
            return self.predefined_attractors
        
        template = structure.get_template()
        n_template = len(template)
        atomic_numbers = structure.get_atomic_numbers()
        types = list(set(atomic_numbers))

        if isinstance(self.possible_number_of_attractors, int):
            number_of_attractors = self.possible_number_of_attractors
        else:
            number_of_attractors = np.random.randint(self.possible_number_of_attractors[0], self.possible_number_of_attractors[1])

        features = self.descriptor.get_local_features(structure)
        if not self.attractors_from_template:
            features = features[n_template:]
            atomic_numbers = atomic_numbers[n_template:]

        attractor_types = np.random.choice(types, size = number_of_attractors, replace = True)
        attractors = np.zeros((number_of_attractors, len(features[0])))
        for _, attractor_type in enumerate(attractor_types):
            filter = atomic_numbers == attractor_type
            possible_indices = np.array(range(len(features)))[filter]
            index1, index2 = np.random.choice(possible_indices, size = 2, replace = False)
            attractor = 1/2 * (features[index2] - features[index1]) + features[index1]
            attractors[_, :] = attractor

        return attractors