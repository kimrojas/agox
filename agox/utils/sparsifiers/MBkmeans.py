from warnings import UserWarning
from typing import List, Optional

import numpy as np
from ase import Atoms
from sklearn.cluster import MiniBatchKMeans

from agox.utils.sparsifiers.ABC_sparsifier import SparsifierBaseClass


class MBkmeans(SparsifierBaseClass):
    name = "MBkmeans"

    def __init__(
        self,
        batch_size: int = 1024,
        fast: bool = False,
        exact_points: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        #Throw a warning saying that this cannot be used with other sparsifiers
        UserWarning(
            "MBkmeans cannot be used with other sparsifiers. "
            "Or together with a method that requires indices of selected features"
            "Use k-medoids instead."
        )
        
        self.batch_size = batch_size
        self.fast = fast

        self.cluster = MiniBatchKMeans(
            n_clusters=self.m_points,
            batch_size=self.batch_size,
            random_state=seed,
            init="k-means++",
            n_init=3,
        )

    def sparsify(
        self, atoms: Optional[List[Atoms]] = None, X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        X = self.preprocess(atoms, X)

        if X.shape[0] < self.m_points:
            m_indices = np.arange(0, X.shape[0])
            return X, m_indices

        if hasattr(self.cluster, "cluster_centers_") and self.fast:
            self._MB_episode(X)
        else:
            self.cluster.fit(X)

        if self.exact_points:
            dists = self.cluster.transform(X)
            min_idx = np.argmin(dists, axis=0)
            Xm = X[min_idx, :]
        else:
            Xm = self.cluster.cluster_centers_

        return Xm, None

    def _MB_episode(self, X: np.ndarray) -> None:
        n_samples = X.shape[0]
        batch_size = min(self.batch_size, n_samples)
        steps = n_samples // batch_size
        random_state = np.random.RandomState()
        for _ in range(steps):
            minibatch_indices = random_state.randint(0, n_samples, batch_size)
            self.cluster.partial_fit(X[minibatch_indices])
