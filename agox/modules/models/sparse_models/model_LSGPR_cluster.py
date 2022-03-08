import numpy as np
from agox.modules.models import LSGPRModel

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


class LSGPRModelCluster(LSGPRModel):
    name = 'LSGPRModelCluster'

    def __init__(self, cluster_method='KMeans', m_distribution=None, include_best=True, include_transfer=False, **kwargs):
        """ m_distribution must be given as dict of energy:number e.g. {0.5:100, 1:100, 2:50, 50:50}
        """
        super().__init__(**kwargs)
        self.include_best = include_best
        self.include_transfer = include_transfer
        if m_distribution is None:
            self.m_distribution = {9999:self.m_points}
        else:
            self.m_distribution = m_distribution

        
        if cluster_method is 'KMeans':
            self.cluster = KMeans
        elif cluster_method is 'agglomerative':
            self.cluster = AgglomerativeClustering
        else:
            print('Cluster method is not known: use either KMeans or agglomerative')
            print('Using KMeans as default!')
            self.cluster = KMeans
        
    def _train_sparse(self, atoms_list):
        m_indices = []
        if self.include_transfer and len(self.transfer_data)>0:
            num_transfer_env = np.sum([len(a) for a in self.transfer_data])
            X = self.Xn[int(num_transfer_env):, :]
            ys = self.L.T @ self.y[len(self.transfer_data):].reshape(-1,1)
            m_indices += np.arange(X.shape[0], self.Xn.shape[0]).tolist()
        else:
            X = self.Xn
            ys = self.L.T @ self.y.reshape(-1,1)

        
        if self.include_best and len(self.transfer_data)>0:
            best_idx = np.argmin(self.y[len(self.transfer_data):])
            indices, = np.nonzero(self.L[best_idx, :])
            m_indices += indices.tolist()

        min_e = 0
        print(f'Total number of local environments: {len(ys)}')
        for max_e, n_clusters in self.m_distribution.items():
            indices = self._get_X_in_energy_interval(ys, min_e, max_e)
            if len(indices)<=n_clusters:
                print('Total number of local environments is less than number of clusters!')
                m_indices = indices
                continue
            cluster = self.cluster(n_clusters=np.amin([n_clusters,len(indices)])).fit(X[indices,:])
            labels = cluster.labels_
            
            print(f'n_clusters={n_clusters}, minE={min_e}, maxE={max_e}, num chosen: {len(np.unique(labels))}, num in interval: {len(indices)}')
            
            for n in range(n_clusters):
                if sum(labels == n)>0:
                    best_idx = np.argmin(ys[indices][labels == n])
                    m_indices.append(np.array(indices)[labels == n][best_idx])
            min_e = max_e

        m_indices = np.unique(m_indices)
        print(f'Number of local environments chosen: {len(m_indices)}')
        self.Xm = self.Xn[m_indices, :]
        return True
            

    def _get_X_in_energy_interval(self, ys, y_min, y_max):
        y = (ys - np.amin(ys)).flatten()
        indices = np.argwhere((y >= y_min) & (y < y_max)).flatten().tolist()
        return indices


    @classmethod
    def default_model(cls, species, single_atom_energies):
        """ 
        - species must be list of symbols or numbers
        - single_atom_energies must be list of floats corresponding to ordering in species or array of 0's
        except at positions of numbers corresponding to species.
        """
        from agox.modules.models.descriptors.soap import SOAP
        from sklearn.gaussian_process.kernels import Exponentiation, DotProduct, ConstantKernel as C
        from ase.atoms import symbols2numbers
        
        if len(species) == len(single_atom_energies):
            sae = np.zeros(100)
            numbers = symbols2numbers(species)
            for num, e in zip(numbers,single_atom_energies):
                sae[num] = e
        else:
            sae = single_atom_energies
            
        descriptor = SOAP(species, r_cut=3, nmax=3, lmax=2, sigma=0.5, normalize=True, use_radial_weighting=False)
        kernel = C(0.005)*Exponentiation(DotProduct(sigma_0=0.), 4)
        return cls(kernel=kernel, descriptor=descriptor, single_atom_energies=sae, noise=0.02)
