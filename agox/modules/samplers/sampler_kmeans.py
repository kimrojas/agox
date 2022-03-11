from argparse import ArgumentDefaultsHelpFormatter
from ase.io import write
from .sampler_ABC import SamplerBaseClass
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class SamplerKMeans(SamplerBaseClass):
    name = 'SamplerKMeans'
    parameters = {}

    def __init__(self, feature_calculator, database=None, model_calculator=None, sample_size=10, max_energy=5, 
                        use_saved_features=False, verbose=False):
        super().__init__()
        self.feature_calculator = feature_calculator
        self.sample_size = sample_size
        self.max_energy = max_energy
        self.verbose = verbose
        self.sample = []
        self.sample_features = []
        self.model_calculator = model_calculator
        self.use_saved_features = use_saved_features
        self.debug = False
        self.database = database

    def setup(self):

        if len(self.database) < 1:
            self.sample = []
            return

        all_finished_structures = self.database.get_all_candidates()
        if len(all_finished_structures) < 1:
            return
        
        e_all = np.array([s.get_potential_energy() for s in all_finished_structures])
        e_min = min(e_all)

        for i in range(5):
            filt = e_all <= e_min + self.max_energy * 2**i
            if np.sum(filt) >= 2*self.sample_size:
                break
        else:
            filt = np.ones(len(e_all), dtype=bool)
            index_sort = np.argsort(e_all)
            filt[index_sort[2*self.sample_size:]] = False

        structures = [all_finished_structures[i] for i in range(len(all_finished_structures)) if filt[i]]
        e = e_all[filt]
        #f = np.array(self.feature_calculator.get_featureMat(structures))
        f = self.get_features(structures)

        n_clusters = 1 + min(self.sample_size-1, int(np.floor(len(e)/5)))

        kmeans = KMeans(n_clusters=n_clusters).fit(f)
        labels = kmeans.labels_

        indices = np.arange(len(e))
        sample = []
        sample_features = []
        for n in range(n_clusters):
            filt_cluster = labels == n
            cluster_indices = indices[filt_cluster]
            min_e_index = np.argmin(e[filt_cluster])
            index_best_in_cluster = cluster_indices[min_e_index]
            sample.append(structures[index_best_in_cluster])
            sample_features.append(f[index_best_in_cluster])

        sample_energies = [t.get_potential_energy() for t in sample]
        sorting_indices = np.argsort(sample_energies)

        self.sample = [sample[i] for i in sorting_indices]
        self.sample_features = [sample_features[i] for i in sorting_indices]
        

        sample_energies = [t.get_potential_energy() for t in self.sample]
        print('SAMPLE_DFT_ENERGY  ', '[',','.join(['{:8.3f}'.format(e) for e in sample_energies]),']')
        

        if self.model_calculator is not None and self.model_calculator.ready_state:
            for s in self.sample:
                t = s.copy()
                t.set_calculator(self.model_calculator)
                E = t.get_potential_energy()
                sigma = self.model_calculator.get_property('uncertainty')
                s.add_meta_information('model_energy',E)
                s.add_meta_information('uncertainty',sigma)
            print('SAMPLE_MODEL_ENERGY', \
                  '[',','.join(['{:8.3f}'.format(t.get_meta_information('model_energy')) for t in self.sample]),']')
            print('SAMPLE_MODEL_SIGMA', \
                  '[',','.join(['{:8.3f}'.format(t.get_meta_information('uncertainty')) for t in self.sample]),']')

        if self.verbose:
            if self.debug:
                write(f'all_strucs_episode_{self.get_episode_counter()}.traj', all_finished_structures)
                write(f'filtered_strucs_episode_{self.get_episode_counter()}.traj', structures)
            write(f'sample_episode_{self.get_episode_counter()}.traj', self.sample)
         
    def assign_to_closest_sample(self,candidate_object):

        if len(self.sample) == 0:
            return None
        
        # find out what cluster we belong to
        f_this = self.feature_calculator.get_featureMat([candidate_object])
        distances = cdist(f_this, self.sample_features, metric='euclidean').reshape(-1)
        if self.verbose:
            print('cdist [',','.join(['{:8.3f}'.format(e) for e in distances]),']')

        d_min_index = np.argmin(distances)

        cluster_center = self.sample[d_min_index]
        cluster_center.add_meta_information('index_in_sampler_list_used_for_printing_purposes_only',d_min_index)
        return cluster_center

    def adjust_sample_cluster_distance(self, chosen_candidate, closest_sample):
        # See if this structure had a lower energy than the lowest energy in the cluster it belongs to
        chosen_candidate_energy = chosen_candidate.get_potential_energy()
        closest_sample_energy = closest_sample.get_potential_energy()
        d_min_index = closest_sample.get_meta_information('index_in_sampler_list_used_for_printing_purposes_only')
        print('CLUST_RES {:06d}'.format(self.get_episode_counter()),
              '[',','.join(['{:8.3f}'.format(t.get_potential_energy()) for t in self.sample]),
              '] {:8.3f} {:8.3f}'.format(closest_sample_energy,chosen_candidate_energy),
              'NEW {:d} {}'.format(d_min_index,chosen_candidate.get_meta_information('description')) \
              if closest_sample_energy > chosen_candidate_energy else '')

    def assign_from_main(self, main):
        super().assign_from_main(main)
        self.get_episode_counter = main.get_episode_counter

    def get_features(self, structures):
        if self.use_saved_features:
            features = []
            for candidate in structures:
                F = candidate.get_meta_information('kmeans_feature')
                if F is None:
                    F = self.feature_calculator.get_feature(candidate)
                    candidate.add_meta_information('kmeans_feature', F)
                features.append(F)
            features = np.array(features)
        else:
            features = np.array(self.feature_calculator.get_featureMat(structures))
        return features
