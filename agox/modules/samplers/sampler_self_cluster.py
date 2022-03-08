from .sampler_ABC import SamplerBaseClass
import numpy as np
from ase.io import write
from scipy.spatial.distance import cdist

class SamplerSelfCluster(SamplerBaseClass):

    
    name = 'SamplerSelfCluster'

    def __init__(self, feature_calculator, model_calculator, database=None, sample_cluster_dist=0.5,sample_cluster_energy_span=3,sample_size=10,verbose=False,debug=False):
        super().__init__(**kwargs)
        self.sample_cluster_dist = sample_cluster_dist
        self.sample_cluster_energy_span = sample_cluster_energy_span
        self.sample_size = sample_size
        self.verbose = verbose
        self.debug = debug
        self.feature_calculator = feature_calculator
        self.model_calculator = model_calculator

        assert database is not None
        self.new_memory = database
        self.sample = []

    def setup(self):
        if self.get_episode_counter() % 100 == 0:
            self.sample_cluster_dist /= 3

        all_finished_structures = self.new_memory.get_all_candidates()
        if self.debug:
            write('all_finished_structures_{:06d}.traj'.format(self.get_episode_counter()),all_finished_structures)
        if self.verbose:
            print('len(all_finished_structures):',len(all_finished_structures))
        if len(all_finished_structures) == 0:
            return
        
        all_finished_structures.sort(key=lambda x: x.get_potential_energy())

        self.f_all = self.feature_calculator.get_featureMat(all_finished_structures)

        # Filter criteria
        dist_threshold = self.sample_cluster_dist
        print('SAMPLE_THRESHOLD',dist_threshold)

        # Apply filtering
        self.sample_indices = [0]
        self.sample_consists_of_lowest_structures = True
        for i, f_i in enumerate(self.f_all):
            distances = cdist(f_i.reshape(1,-1), self.f_all[self.sample_indices], metric='euclidean').reshape(-1)
            d_min = np.min(distances)
            if d_min > dist_threshold:
                self.sample_indices.append(i)
                print('SAMPLE index',i,d_min)
            elif i > 0:
                self.sample_consists_of_lowest_structures = False
            if len(self.sample_indices) >= self.sample_size:
                break
        
        self.sample = [all_finished_structures[i] for i in self.sample_indices]
        if self.verbose:
            write('sample_{:06d}.traj'.format(self.get_episode_counter()),self.sample)
        sample_energies = [t.get_potential_energy() for t in self.sample]
        print('SAMPLE_DFT_ENERGY  ', '[',','.join(['{:8.3f}'.format(e) for e in sample_energies]),']',
              self.sample_consists_of_lowest_structures)

        if hasattr(self.model_calculator,'ever_trained') and self.model_calculator.ever_trained:
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

    def assign_to_closest_sample(self,candidate_object):

        if len(self.sample) == 0:
            return None
        
        # find out what cluster we belong to
        f_this = self.feature_calculator.get_featureMat([candidate_object])
        distances = cdist(f_this, self.f_all[self.sample_indices], metric='euclidean').reshape(-1)
        if self.verbose:
            print('cdist [',','.join(['{:8.3f}'.format(e) for e in distances]),']')
        # assume that this is a new cluster that should be compared in energy to the highest one so far
        d_min_index = len(self.sample_indices) - 1
        # run through existing clusters in ascending energy order and stop first time we belong to one
        #   if no match, we already were assigned to the highest energy one
        for li in range(d_min_index):
            if distances[li] <= self.sample_cluster_dist:
                d_min_index = li
                break
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

        # adjust the cluster distance hyper parameter
        if chosen_candidate_energy - closest_sample_energy < 0:
            if not self.sample_consists_of_lowest_structures:
                self.sample_cluster_dist *= 0.95
                print('CLUST_DISTRIB_IMPROVED,  sample_cluster_dist: {:8.3f}'.format(self.sample_cluster_dist))
        else:
            if len(self.sample) == 1:
                self.sample_cluster_dist *= 0.95
            else:
                energy_span = self.sample[-1].get_potential_energy() - self.sample[0].get_potential_energy()
                if energy_span < self.sample_cluster_energy_span:
                    self.sample_cluster_dist *= 1.05
                print('CLUST_DISTRIB_UNCHANGED, sample_cluster_dist: {:8.3f}, energy span: {:8.3f}'.format(self.sample_cluster_dist,energy_span))

    def assign_from_main(self, main):
        super().assign_from_main(main)
