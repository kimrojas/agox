from typing import List
from warnings import warn

import numpy as np
from ase import Atoms

from agox.models.descriptors.ABC_descriptor import DescriptorBaseClass
from agox.utils.filters.ABC_filter import FilterBaseClass
from agox.utils.sparsifiers.ABC_sparsifier import SparsifierBaseClass


class Filter(FilterBaseClass):
    def __init__(
        self,
        sparsifier: SparsifierBaseClass,
        descriptor: DescriptorBaseClass = None,
        descriptor_type: str = "global",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sparsifier = sparsifier
        if descriptor is not None:
            self.descriptor = descriptor
            self.feature_method = getattr(
                self.descriptor, "get_" + descriptor_type + "_features"
            )
        else:
            warn("Using indexes as features")
            self.feature_method = lambda atoms: np.arange(len(atoms))

    def _filter(self, atoms: List[Atoms]) -> np.ndarray:
        X = self.preprocess(atoms)
        _, idxs = self.sparsifier(X)
        return idxs

    def preprocess(self, atoms: List[Atoms] = None) -> np.ndarray:
        """
        Preprocess the data by computing the features.

        Parameters
        ----------
        atoms : List[Atoms]
            List of atoms objects

        Returns
        -------
        X : np.ndarray
            Matrix with rows corresponding to features

        """

        f = self.feature_method(atoms)
        if isinstance(f, np.ndarray) and len(f.shape) == 1:
            f = f.reshape(1, -1)
        f = np.vstack(f)
        return f


    @property
    def name(self):
        return f"{self.sparsifier.name}-Filter"
