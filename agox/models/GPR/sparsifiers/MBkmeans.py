import numpy as np
from agox.models.GPR.sparsifiers.ABC_sparsifier import SparsifierBaseClass
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.validation import check_random_state


class MBkmeans(SparsifierBaseClass):
    
    name = 'MBkmeans'

    def __init__(self, m_points, batch_size=1024, fast=False,
                 exact_points=False, seed=None, **kwargs):
        super().__init__(m_points, **kwargs)

        self.batch_size = batch_size
        self.fast = fast
        self.full_update_interval = full_update_interval
        self.cluster_weights = cluster_weights

        self.cluster = MiniBatchKMeans(n_clusters=self.m_points, batch_size=self.batch_size, random_state=seed,
                                       init='k-means++', n_init=3)

        
    def sparsify(self, X):
        if X.shape[0] < self.m_points:
            m_indices = np.arange(0, X.shape[0])
            return X, m_indices

        if hasattr(self.cluster, 'cluster_centers_') and self.fast:
            self._MB_episode(X)
        else:
            self.cluster.fit(X)

        if self.exact_points:
            dists = self.cluster.transform(X)
            min_idx = np.argmin(dists, axis=0)
            Xm = X[min_idx, :]
            return Xm, min_idx
        else:
            Xm = self.cluster.cluster_centers_
        
        return Xm, None
            

    def _MB_episode(self, X):
        n_samples = X.shape[0]
        batch_size = min(self.batch_size, n_samples)
        steps = n_samples // batch_size
        random_state = np.random.RandomState()
        for _ in range(steps):
            minibatch_indices = random_state.randint(0, n_samples, batch_size)
            self.cluster.partial_fit(X[minibatch_indices])
            


