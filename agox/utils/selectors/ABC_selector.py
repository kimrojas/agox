from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms


class SelectorBaseClass(ABC):
    """
    Abstract class for selectors.

    A selector is a class that takes a list of atoms objects and returns a
    subset of the atoms objects and a corresponding list of indices.

    Posible inputs are:
        - atoms: a list of atoms objects
        - features: a list of feature vectors
        - indices: a list of indices

    Possible outputs are:
        - atoms: a list of atoms objects
        - features: a list of feature vectors
        - indices: a list of indices

    Methods
    -------
    select(atoms: List[Atoms], **kwargs) -> Tuple[Union[List[Atoms], np.ndarray], np.ndarray]
        Returns a subset of the atoms objects or a feature matrix and a corresponding list of indices.

    Arguments
    ---------
    output: str
        The output of the selector. Must be either "atoms" or "features".

    """

    def __init__(
        self,
        input=["indices", "atoms, features"],
        output=["indices", "atoms", "features"],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.output = []
        self.input = []

        for i in input:
            if i in self.implemented_inputs:
                self.input.append(i)
            else:
                raise ValueError(f"Input {i} not implemented.")

        assert set(self.required_inputs).issubset(
            set(self.input)
        ), "Missing required inputs."

        for o in output:
            if o in self.implemented_outputs:
                self.output.append(o)
            else:
                raise ValueError(f"Output {o} not implemented.")

        # check that the output is a subset of the input
        assert set(self.output).issubset(
            set(self.input)
        ), "Output must be a subset of the input."
        
    @abstractmethod
    def select(
        self,
        indices: Optional[np.ndarray] = None,
        atoms: Optional[List[Atoms]] = None,
        X: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Tuple[Optional[np.ndarray], Optional[List[Atoms]], Optional[np.ndarray]]:
        """
        Selects a subset of the inputs and returns the subset and the corresponding outputs

        Parameters
        ----------
        indices: Optional[np.ndarray]
            The indices of the atoms objects to be selected.
        atoms: Optional[List[Atoms]]
            The atoms objects to be selected.
        X: Optional[np.ndarray]
            The feature matrix to be selected.

        Returns
        -------
        selected_idxs : Optional[np.ndarray]
            Indices of selected atoms.

        atoms : Optional[List[Atoms]]
            Subset of the input Atoms-like objects or a feature matrix

        features : Optional[np.ndarray]
            Subset of the input Atoms-like objects or a feature matrix

        """

        pass

    @property
    @abstractmethod
    def implemented_input(self) -> List[str]:
        """Returns the implemented input types"""
        pass

    @property
    @abstractmethod
    def required_input(self) -> List[str]:
        """Returns the required input types"""
        pass

    @property
    @abstractmethod
    def implemented_output(self) -> List[str]:
        """Returns the implemented output types"""
        pass

    @property
    @abstractmethod
    def name(self):
        return NotImplementedError

    def __call__(
        self, **kwargs
    ) -> Tuple[Optional[np.ndarray], Optional[List[Atoms]], Optional[np.ndarray]]:
        return self.select(**kwargs)

    def __add__(self, other: "SelectorBaseClass") -> "SelectorBaseClass":
        return SelectorSum(s0=self, s1=other, input=self.input, output=other.output)


class SelectorSum(SelectorBaseClass):
    def __init__(self, s0: SelectorBaseClass, s1: SelectorBaseClass, **kwargs) -> None:
        super().__init__(**kwargs)

        self.s0 = s0
        self.s1 = s1

        print("=== Checking selector input and output compatibility ===")
        # check that the selectors have compatible inputs and outputs
        assert (
            self.s1.input == self.s0.output
        ), "Selectors have mismatching inputs and outputs"

        # check that the input of s1 is a subset of the input of s0
        assert set(self.s1.output).issubset(
            set(self.s0.input)
        ), "The output of the second selector must be a subset of the input of the first selector."

        # check that the output of s1 is a subset of the output of s0
        assert set(self.s1.output).issubset(
            set(self.s0.output)
        ), "The output of the second selector must be a subset of the output of the first selector."

        print("=== Input and output compatibility check passed ===")

    def select(
        self, **kwargs
    ) -> Tuple[Optional[np.ndarray], Optional[List[Atoms]], Optional[np.ndarray]]:
        # select with s0
        selected_idxs0, atoms, features = self.s0.select(**kwargs)

        # select with s1
        selected_idxs1, atoms, features = self.s1.select(
            indices=selected_idxs0, atoms=atoms, features=features
        )

        if "indices" in self.output:
            selected_idxs = selected_idxs0[selected_idxs1]
        else:
            selected_idxs = None

        return selected_idxs, atoms, features

    @property
    def implemented_input(self) -> List[str]:
        return ["indices", "atoms", "features"]

    @property
    def implemented_output(self) -> List[str]:
        return ["indices", "atoms", "features"]

    @property
    def required_input(self) -> List[str]:
        return []

    @property
    def name(self):
        return "+".join([selector.name for selector in self.selectors])

