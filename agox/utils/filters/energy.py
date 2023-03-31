from typing import List

import numpy as np
from ase import Atoms

from agox.utils.filters.ABC_filter import FilterBaseClass


class EnergyFilter(FilterBaseClass):
    """
    Filter that removes all structures with energy above a certain threshold.

    """

    name = "EnergyFilter"

    def __init__(self, delta_E: float = 1000, **kwargs):
        """
        Parameters
        ----------
        max_energy: float
            Maximum energy above minimum energy.
        """
        super().__init__(**kwargs)
        self.delta_E = delta_E

    def _filter(self, atoms: List[Atoms]) -> np.ndarray:
        Es = np.array([a.get_potential_energy() for a in atoms])
        E_min = np.min(Es)
        E_boundary = E_min + self.delta_E
        # get indicies where Es is greater than E_boundary
        indexes = np.array([i for i, E in enumerate(Es) if E < E_boundary])

        return indexes
