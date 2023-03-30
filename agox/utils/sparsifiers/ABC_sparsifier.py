from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from ase import Atoms

from agox.models.descriptors import DescriptorBaseClass
from agox.utils.filters import FilterBaseClass


class SparsifierBaseClass(ABC):
    """
    Abstract class for sparsifiers.

    A sparsifier is a function that takes a matrix X with rows corresponding to feaures.
    It returns a matrix with the same number of columns, possibly together with which rows are selected.


    Attributes
    ----------
    m_points : int
        Number of points to select

    Methods
    -------
    sparsify(atoms: List[Atoms] X): np.ndarray) -> np.ndarray:
        Sparsify the data
    """

    def __init__(
        self,
        descriptor: DescriptorBaseClass,
        m_points: int = 1000,
        descriptor_type: str = "global",
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        descriptor : DescriptorBaseClass
            Descriptor to use for computing the features
        m_points : int
            Number of points to select
        descriptor_type : str
            Type of descriptor to use. Can be "global" or "local"

        """
        super().__init__(**kwargs)
        self.m_points = m_points
        self.descriptor = descriptor
        self.feature_method = getattr(
            self.descriptor, "get_" + descriptor_type + "_features"
        )

    @abstractmethod
    def sparsify(
        self, atoms: Optional[List[Atoms]] = None, X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns a matrix with the same number of columns together with which rows are selected.

        Parameters
        ----------
        atoms : List[Atoms]
            List of atoms objects
        X : np.ndarray
            Matrix with rows corresponding to features

        Returns
        -------
        X_sparsified : array-like, shape (m_points, n_samples)
            Matrix with rows corresponding to feaures.


        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        return NotImplementedError

    def preprocess(
        self, atoms: Optional[List[Atoms]] = None, X: Optional[np.ndarray] = None
    ) -> np.ndarray:
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
        if X is None:
            assert atoms is not None, "Either atoms or X must be provided"
            X = self.feature_method(atoms)
        return X

    def __call__(
        self, atoms: Optional[List[Atoms]] = None, X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return self.sparsify(X)

    def __add__(self, other: "SparsifierBaseClass"):
        return SumSparsifier(s0=self, s1=other)


class SumSparsifier(SparsifierBaseClass):
    """
    Sum of two sparsifiers or a filter and a sparsifier.

    Attributes
    ----------
    s0 : Union[SparsifierBaseClass, FilterBaseClass]
        First sparsifier or filter
    s1 : SparsifierBaseClass
        Second sparsifier

    """

    def __init__(
        self,
        s0: Union[SparsifierBaseClass, FilterBaseClass],
        s1: SparsifierBaseClass,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.s0 = s0
        self.s1 = s1

        if isinstance(s0, SparsifierBaseClass):
            self.output = "X"
            # assert that the descriptor is the same
            assert s0.descriptor == s1.descriptor, "Descriptors must be the same"

        elif isinstance(s0, FilterBaseClass):
            self.output = "atoms"
        else:
            raise ValueError(
                "Cannot sum something that is not a sparsifier or a filter"
            )

        # assert that s1 is not a filter
        if isinstance(s1, FilterBaseClass):
            raise ValueError("Cannot do: sparsifier + filter!")

    def sparsify(
        self, atoms: Optional[List[Atoms]] = None, X: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Returns a matrix with the same number of columns together with which rows are selected.

        Parameters
        ----------
        atoms : List[Atoms]
            List of atoms objects
        X : np.ndarray
            Matrix with rows corresponding to features

        Returns
        -------
        X_sparsified : array-like, shape (m_points, n_features)
            Matrix with rows corresponding to feaures.
        """
        # select with s0
        out = self.s0(atoms=atoms, X=X)

        if self.output == "X":
            return self.s1(X=out)
        elif self.output == "atoms":
            return self.s1(atoms=out[0])
        else:
            raise ValueError(
                "Cannot sum something that is not a sparsifier or a filter"
            )

    @property
    def name(self):
        return f"{self.s0.name}+{self.s1.name}"
