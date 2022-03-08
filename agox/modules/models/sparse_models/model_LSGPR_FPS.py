import numpy as np
from agox.modules.models import LSGPRModel
from scipy.spatial.distance import cdist

class LSGPRModelFPS(LSGPRModel):
    name = 'LSGPRModelFPS'

    def __init__(self, atom_distribution=None, include_best=True, include_transfer=False, **kwargs):
        """ atom_distribution must be given as dict of atomic number : number e.g. {0.5:100, 1:100, 2:50, 50:50}
        """
        super().__init__(**kwargs)
        self.include_best = include_best
        self.include_transfer = include_transfer
        self.atom_distribution = atom_distribution
        
    

    def _train_sparse(self, atoms_list):
        print('number of local points', self.Xn.shape[0])
        if self.m_points > self.Xn.shape[0]:
            self.Xm = self.Xn
            return True
        
        m_indices = []
        X = self.Xn
        
        if self.include_best and len(self.transfer_data)>0:
            best_idx = np.argmin(self.y[len(self.transfer_data):])
            indices, = np.nonzero(self.L[best_idx, :])
            m_indices += indices.tolist()
            self.Xm = self.Xn[m_indices, :]
            
            mask_rest = np.ones(self.Xn.shape[0], dtype=bool)
            mask_rest[m_indices] = False
            X = X[mask_rest, :]
        else:
            self.Xm = np.array([])
        
        
        new_index = np.random.randint(0, high=X.shape[0])
        if len(self.Xm) == 0:
            self.Xm = X[new_index, :].reshape(1,-1)
        else:
            self.Xm = np.vstack([self.Xm, X[new_index, :]])
        X = np.delete(X, new_index, axis=0)
        print('X shape', X.shape)
        print('Xm', self.Xm[-1,:])
        while self.Xm.shape[0] < self.m_points:
            distances = cdist(self.Xm[-1,:].reshape(1,-1), X).reshape(-1)
            new_index = np.argmax(distances)
            self.Xm = np.vstack([self.Xm, X[new_index, :]])
            X = np.delete(X, new_index, axis=0)

        return True
            

    def _get_indices_with_atomic_number(self, atoms_list, atomic_number):
        
        return indices


    @classmethod
    def default_model(cls, species, single_atom_energies):
        """ 
        - species must be list of symbols or numbers
        - single_atom_energies must be list of floats corresponding to ordering in species or array of 0's
        except at positions of numbers corresponding to species.
        """
        from agox.modules.models.descriptors.soap import SOAP
        from sklearn.gaussian_process.kernels import Exponentiation, DotProduct, ConstantKernel as C
        from ase.atoms import symbols2numbers
        
        if len(species) == len(single_atom_energies):
            sae = np.zeros(100)
            numbers = symbols2numbers(species)
            for num, e in zip(numbers,single_atom_energies):
                sae[num] = e
        else:
            sae = single_atom_energies
            
        descriptor = SOAP(species, r_cut=3, nmax=3, lmax=2, sigma=0.5, normalize=True, use_radial_weighting=False)
        kernel = C(0.005)*Exponentiation(DotProduct(sigma_0=0.), 4)
        return cls(kernel=kernel, descriptor=descriptor, single_atom_energies=sae, noise=0.02)
