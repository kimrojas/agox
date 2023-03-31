from typing import List
import warnings
from warnings import UserWarning

import numpy as np
from ase import Atoms

from agox.models.descriptors.ABC_descriptor import DescriptorBaseClass
from agox.utils.filters.ABC_filter import FilterBaseClass
from agox.utils.sparsifiers.ABC_sparsifier import SparsifierBaseClass


class Filter(FilterBaseClass):
    def __init__(
        self,
        sparsifier: SparsifierBaseClass,
        descriptor: DescriptorBaseClass,
        descriptor_type: str = "global",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if descriptor is not None:
            self.descriptor = descriptor
            self.feature_method = getattr(
                self.descriptor, "get_" + descriptor_type + "_features"
            )
        else:
            UserWarning("Using indexes as features")
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

        return self.feature_method(atoms)
