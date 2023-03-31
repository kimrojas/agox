from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from ase import Atoms


class FilterBaseClass(ABC):
    """Base class for filters.

    This class is used to define the base class for filters. All filters
    should inherit from this class and implement the methods defined here.

    Attributes
    ----------

    Methods
    -------
    _filter(atoms: List[Atoms]) -> np.ndarray:
        Filter the atoms.

    __call__(atoms: List[Atoms]) -> Tuple[List[Atoms], np.ndarray]:
        Filter the atoms and return the selected atoms and indexes of the filtered atoms.

    """

    def __init__(self, **kwargs):
        """Initialize the filter."""
        super().__init__(**kwargs)

    def filter(self, atoms: List[Atoms]) -> Tuple[List[Atoms], np.ndarray]:
        indexes = self._filter(atoms)
        return [atoms[i] for i in indexes], indexes

    @abstractmethod
    def _filter(self, atoms: List[Atoms]) -> Tuple[List[Atoms], np.ndarray]:
        """Filter the atoms object.

        Parameters
        ----------
        atoms
            The atoms object to be filtered.

        Returns
        -------
        indexes: array-like
            The indexes of the atoms that are kept.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        return NotImplementedError

    def __call__(self, atoms: List[Atoms]) -> List[Atoms]:
        return self.filter(atoms)

    def __add__(self, other: "FilterBaseClass"):
        return SumFilter(f0=self, f1=other)


class SumFilter(FilterBaseClass):
    """Sum filter.

    This class implement to sum of two filters.

    Attributes
    ----------
    f0 : FilterBaseClass
        The first filter.
    f1 : FilterBaseClass
        The second filter.

    """

    def __init__(self, f0: FilterBaseClass, f1: FilterBaseClass, **kwargs):
        """Initialize the filter."""
        super().__init__(**kwargs)
        self.f0 = f0
        self.f1 = f1

    def _filter(self, atoms: List[Atoms]) -> List[Atoms]:
        """Filter the atoms object.

        Parameters
        ----------
        atoms
            The atoms object to be filtered.

        Returns
        -------
        List[Atoms]
            The filtered atoms object.
        """
        f0_atoms, idx0 = self.f0(atoms)
        _, idx1 = self.f1(f0_atoms)
        return idx0[idx1]

    @property
    def name(self) -> str:
        return f"{self.f0.name} + {self.f1.name}"
