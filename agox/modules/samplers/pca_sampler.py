from .sampler_ABC import SamplerBaseClass
import numpy as np
from ase.io import write

from agox.modules.models.gaussian_process.featureCalculators_multi.angular_fingerprintFeature_cy import Angular_Fingerprint
from sklearn.decomposition import PCA

class PCASampler(SamplerBaseClass):

    
    name = 'PCASampler'

    def __init__(self, model_calculator, database=None, collector=None, sample_size=10, verbose=False, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.sample_size = sample_size
        self.verbose = verbose
        self.debug = debug
        self.model_calculator = model_calculator
        self.sample = []

        assert database is not None
        self.new_memory = database
        assert collector is not None
        self.collector = collector


    def setup(self):

        structures = self.collector.all_candidates_ever
        f = [s.get_meta_information('Standard Angular Fingerprint') for s in structures]

        if self.debug:
            write('all_candidate_structures_{:06d}.traj'.format(self.get_episode_counter()),structures)
        if self.verbose:
            print('len(structures):',len(structures))
        if len(structures) < 2: # since the PCA is two-dimensional
            return

        temp_atoms = structures[0]
        feature_calculator = Angular_Fingerprint(temp_atoms,Rc1=6,Rc2=4,binwidth1=0.2,Nbins2=30,
                                                 sigma1=0.2,sigma2=0.2,gamma=2,eta=20,use_angular=True)

        # Calculate features
        #f = feature_calculator.get_featureMat(structures)

        # Fit PCA
        pca = PCA(n_components=2)
        pca.fit(f)

        # Project data to 2D with PCA
        f2d = pca.transform(f)

        xmax = max(f2d[:,0])
        xmin = min(f2d[:,0])
        xspan = xmax - xmin
        xbins = np.linspace(xmin,xmax + 1e-8 * xspan,11)[1:]
        xbin_indices = np.digitize(f2d[:,0],xbins)

        ymax = max(f2d[:,1])
        ymin = min(f2d[:,1])
        yspan = ymax - ymin
        ybins = np.linspace(ymin,ymax + 1e-8 * yspan,11)[1:]
        ybin_indices = np.digitize(f2d[:,1],ybins)

        print('PCA1: {:8.3f}'.format(xspan),xmin,xmax)
        print('PCA2: {:8.3f}'.format(yspan),ymin,ymax)

        structures_for_sample = []
        for j in range(len(ybins)):
            for i in range(len(xbins)):
                relevant_ones = (xbin_indices==i) & (ybin_indices==j)
                ks = np.arange(len(structures))[relevant_ones]
                if len(ks) == 0:
                    continue
                strucs = [structures[k] for k in ks]
                sorted_strucs = sorted(strucs, key=lambda x: x.get_potential_energy())
                structures_for_sample.append(sorted_strucs[0])

        self.sample = sorted(structures_for_sample, key=lambda x: x.get_potential_energy())

        if self.verbose:
            write('sample_{:06d}.traj'.format(self.get_episode_counter()),self.sample)
        sample_energies = [t.get_potential_energy() for t in self.sample]
        print('SAMPLE_DFT_ENERGY  ', '[',','.join(['{:8.3f}'.format(e) for e in sample_energies]),']')

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
        
        i = 0

        cluster_center = self.sample[i]
        cluster_center.add_meta_information('index_in_sampler_list_used_for_printing_purposes_only',i)
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
        self.model_calculator.assign_from_main(main)
