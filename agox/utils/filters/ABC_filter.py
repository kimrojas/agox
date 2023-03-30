from abc import ABC, abstractmethod
from typing import List, Union, Tuple

from ase import Atoms

from agox.utils.sparsifiers.ABC_sparsifier import (SparsifierBaseClass,
                                                   SumSparsifier)


class FilterBaseClass(ABC):
    """Base class for filters.

    This class is used to define the interface for filters. All filters
    should inherit from this class and implement the methods defined here.

    Attributes
    ----------

    Methods
    -------
    filter(atoms: List[Atoms]) -> List[Atoms]
        Filter the atoms.

    """

    def __init__(self, **kwargs):
        """Initialize the filter."""
        super().__init__(**kwargs)

    @abstractmethod
    def filter(self, atoms: List[Atoms]) -> Tuple[List[Atoms], List[Atoms]]:
        """Filter the atoms object.

        Parameters
        ----------
        atoms
            The atoms object to be filtered.

        Returns
        -------
        List[Atoms]
            The selected atoms object by the filter.
        List[Atoms]
            The rest of the atoms object not selected by the filter.
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
        f0, _ = self.f0.filter(atoms)
        return self.f1(f0)

    @property
    def name(self) -> str:
        return f"{self.f0.name} + {self.f1.name}"
