import numpy as np
from ase.data import covalent_radii
from agox.modules.postprocessors.postprocess_ABC import PostprocessBaseClass

class MinimumDistPostProcess(PostprocessBaseClass):

    name = 'MinimumDistance'
    def __init__(self, c1=0.5, **kwargs):
        super().__init__(**kwargs)
        self.c1 = c1
    
    def postprocess(self, candidate):
        if candidate is None:
            return None
        distances = candidate.get_all_distances()
        cov_dist = np.array([covalent_radii[n] for n in candidate.numbers])
        min_distances = self.c1 * np.add.outer(cov_dist, cov_dist)
        if np.any(np.tril(distances-min_distances, -1) < 0):
            return None
        else:
            return candidate


if __name__ == '__main__':
    from ase.build import molecule
    h20 = molecule('H2O')
    min_dist = MinimumDistPostProcess()
    min_dist.postprocess(h20)
        
