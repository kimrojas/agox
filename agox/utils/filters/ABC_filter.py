from abc import ABC, abstractmethod
from typing import List, Union

from ase import Atoms

from agox.utils.sparsifiers.ABC_sparsifier import (SparsifierBaseClass,
                                                   SumSparsifier)


class FilterBaseClass(ABC):
    """Base class for filters.

    This class is used to define the interface for filters. All filters
    should inherit from this class and implement the methods defined here.
    """

    def __init__(self, **kwargs):
        """Initialize the filter."""
        pass

    @abstractmethod
    def filter(self, atoms: List[Atoms]) -> List[Atoms]:
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
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        return NotImplementedError

    def __call__(self, atoms: List[Atoms]) -> List[Atoms]:
        return self.filter(atoms)

    def __add__(self, other: Union["FilterBaseClass", SparsifierBaseClass]):
        if isinstance(other, FilterBaseClass):
            return SumFilter(f0=self, f1=other)
        elif isinstance(other, SparsifierBaseClass):
            return SumSparsifier(s0=self, s1=other)
        else:
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")


class SumFilter(FilterBaseClass):
    """Sum filter.

    This filter adds the number of atoms in the first and second
    atoms object.
    """

    def __init__(self, f0: FilterBaseClass, f1: FilterBaseClass, **kwargs):
        """Initialize the filter."""
        super().__init__(**kwargs)
        self.f0 = f0
        self.f1 = f1

    def filter(self, atoms: List[Atoms]) -> List[Atoms]:
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
        return self.f1(self.f0(atoms))

    @property
    def name(self) -> str:
        return f"{self.f0.name} + {self.f1.name}"
