from select import select
import numpy as np
from agox.modules.acquisitors import AcquisitorBaseClass
from sklearn.cluster import KMeans

class KmeansAcquisitor(AcquisitorBaseClass):

    name = 'KmeansAcquisitor'

    def __init__(self, descriptor, model, k=5, energy_threshold=5, start_iteration=0, **kwargs):
        super().__init__(**kwargs)
        self.descriptor = descriptor
        self.model = model
        self.k = k
        self.start_iteration = start_iteration

        self.energy_threshold = energy_threshold

    def calculate_acquisition_function(self, candidates):    
        pass

    def sort_according_to_acquisition_function(self, candidates):        
        # Calculate features for all candidates:

        if self.get_iteration_counter() >= self.start_iteration and self.model.ready_state: 

            # Filter according to energy:
            [candidate.set_calculator(self.model) for candidate in candidates]
            energies = np.array([a.get_potential_energy() for a in candidates])
            valid_data = np.argwhere(energies <= np.min(energies) + self.energy_threshold).flatten()
            candidates = [candidates[i] for i in valid_data]

            # Determine the number of clusters.. This is mostly a fail safe.
            if not len(candidates) >= self.k:
                k = len(candidates)
            else:
                k = self.k

            # Get features                
            X = np.array([self.descriptor.get_feature(candidate) for candidate in candidates])
            if X.ndim == 3: # (Structure, atoms, features) -- sum away the atoms dimension to get (structure, features)                
                X = np.sum(X, axis=1)

            # Make clustering:
            k_means = KMeans(init="k-means++", n_clusters=k, n_init=10)
            labels = k_means.fit_predict(X)
            centers = k_means.cluster_centers_

            # Selected from clusters
            selected = self.select_from_clustering(candidates, labels, centers)        
            selected_candidates = [candidates[i] for i in selected]
            selected_energies = [atoms.get_potential_energy() for atoms in selected_candidates]
            acquisition_values = [labels[i] for i in selected]

            # Sort selection
            sort_index = np.argsort(selected_energies)
            selected_candidates = [selected_candidates[i] for i in sort_index]

            # Print
            for cand, acq_value in zip(selected_candidates, acquisition_values):
                self.writer(f'Model energy = {cand.get_potential_energy()} - Cluster label = {acq_value}')

            return selected_candidates, acquisition_values
        else:
            return candidates, np.zeros(len(candidates))


    def select_from_clustering(self, candidates, labels, cluster_centers):        
        # Select lowest energy from each cluster: 
        energies = np.array([candidate.get_potential_energy() for candidate in candidates])

        unique_labels = np.unique(labels)

        selected_indices = []

        for unique_label in unique_labels:
            indices = np.argwhere(labels == unique_label)
            sub_index = np.argmin(energies[indices])
            index = indices[sub_index]
            selected_indices.append(index)

        return np.array(selected_indices).flatten()