from typing import List, Tuple

import numpy as np
from ase import Atoms

from agox.utils.filters.ABC_filter import FilterBaseClass


class EnergyFilter(FilterBaseClass):
    """
    Filter that removes all structures with energy above a certain threshold.

    """

    name = "EnergyFilter"

    def __init__(self, delta_E: float = 100, **kwargs):
        """
        Parameters
        ----------
        max_energy: float
            Maximum energy above minimum energy.
        """
        super().__init__(**kwargs)
        self.delta_E = delta_E

    def filter(self, atoms: List[Atoms]) -> Tuple[List[Atoms], List[Atoms]]:
        filtered = []
        indexes = []
        Es = np.array([a.get_potential_energy() for a in atoms])
        E_min = np.min(Es)
        E_boundary = E_min + self.delta_E
        for i, (a, E) in enumerate(zip(atoms, Es)):
            if E < E_boundary:
                filtered.append(a)
                indexes.append(i)

        return filtered, np.array(indexes)
