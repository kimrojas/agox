from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms


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
        m_points: int = 1000,
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

    @abstractmethod
    def sparsify(self, X: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a matrix with the same number of columns together with which rows are selected.

        Parameters
        ----------
        X : np.ndarray
            Matrix with rows corresponding to features

        Returns
        -------
        X_sparsified : array-like, shape (m_points, n_samples)
            Matrix with rows corresponding to feaures.
        selected : array-like, shape (m_points,)

        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        return NotImplementedError

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.sparsify(X)

    def __add__(self, other: "SparsifierBaseClass"):
        return SumSparsifier(s0=self, s1=other)


class SumSparsifier(SparsifierBaseClass):
    """
    Sum of two sparsifiers

    Attributes
    ----------
    s0 : SparsifierBaseClass
        First sparsifier
    s1 : SparsifierBaseClass
        Second sparsifier

    """

    def __init__(
        self,
        s0: SparsifierBaseClass,
        s1: SparsifierBaseClass,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.s0 = s0
        self.s1 = s1

        self.m_points = s1.m_points

    def sparsify(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a matrix with the same number of columns together with which rows are selected.

        Parameters
        ----------
        X : np.ndarray
            Matrix with rows corresponding to features

        Returns
        -------
        X_sparsified : array-like, shape (m_points, n_features)
            Matrix with rows corresponding to feaures.
        selected : array-like, shape (m_points,)
        """
        # select with s0
        Xm0, idxs0 = self.s0(X=X)
        Xm, idxs1 = self.s1(X=Xm0)
        return Xm, idxs0[idxs1]

    @property
    def name(self):
        return f"{self.s0.name}+{self.s1.name}"
