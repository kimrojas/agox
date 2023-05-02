import numpy as np
from agox.generators.complementary_energy.attractor_methods.ABC_attractors import AttractorMethodBaseClass
from sklearn.cluster import KMeans

class AttractorKmeans(AttractorMethodBaseClass):

    name = 'AttractorKmeans'

    def __init__(self, descriptor, number_of_structures = 5, possible_number_of_attractors = [3, 6], attractors_from_template = True, predefined_attractors = None, order = 5):
        super().__init__(order = order)
        self.descriptor = descriptor
        self.number_of_structures = number_of_structures
        self.possible_number_of_attractors = possible_number_of_attractors
        self.attractors_from_template = attractors_from_template
        self.predefined_attractors = predefined_attractors

    def get_attractors(self, structure):

        if self.predefined_attractors is not None:
            return self.predefined_attractors
        
        template = structure.get_template()
        n_template = len(template)

        if isinstance(self.possible_number_of_attractors, int):
            number_of_attractors = self.possible_number_of_attractors
        else:
            number_of_attractors = np.random.randint(self.possible_number_of_attractors[0], self.possible_number_of_attractors[1])

        if len(self.structures) <= self.number_of_structures:
            database_structure_index = range(len(self.structures))
        else:
            database_structure_index = np.random.choice(len(self.structures), size = self.number_of_structures, replace = False)
    
        structures_from_database = [self.structures[i] for i in database_structure_index]

        list_of_features = []
        for struc in structures_from_database:
            features = self.descriptor.get_local_features(struc)
            if not self.attractors_from_template:
                features = features[n_template:]
            list_of_features.append(features)

        list_of_features = np.array(list_of_features).reshape(len(structures_from_database) * len(list_of_features[0]), -1)

        kmeans = KMeans(n_clusters = number_of_attractors).fit(list_of_features)
        attractors = kmeans.cluster_centers_

        return attractors