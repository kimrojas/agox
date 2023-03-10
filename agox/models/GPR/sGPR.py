import warnings
import numpy as np
from scipy.linalg import cholesky, cho_solve, qr, lstsq, LinAlgError
from time import time


from agox.models.GPR.GPR import GPR
from agox.utils import candidate_list_comprehension
from agox.writer import agox_writer
from agox.observer import Observer



class SparseGPR(GPR):

    name = 'SparseGPR'

    implemented_properties = ['energy', 'forces', 'local_energy']

    dynamic_attributes = ['Xm', 'K_inv', 'Kmm_inv', 'alpha']


    """
    Sparse GPR Class

    Attributes
    ----------
    Xn : np.ndarray
        Training data
    Xm : np.ndarray
        Inducing points
    K_mm : np.ndarray
        Kernel matrix between inducing points
    K_nm : np.ndarray
        Kernel matrix between training and inducing points
    K_inv : np.ndarray
        Inverse of K_mm + K_nm.T @ sigma_inv @ K_nm
    Kmm_inv : np.ndarray
        Inverse of K_mm
    L : np.ndarray
        Matrix of ones and zeros indicating which atoms are in which configuration
    

    Methods
    -------
    predict_local_energy(atoms=None, X=None)
        Calculate the local energies in the model.
    
    """
    

    def __init__(self, descriptor, kernel, noise=0.05,
                 centralize=False, jitter=1e-8, **kwargs):

        """

        Parameters
        ----------
        descriptor : DescriptorBaseClass
            Descriptor object
        kernel : KernelBaseClass
            Kernel object
        m_points : int
            Number of inducing points
        transfer_data : list of ase.Atoms
            List of ase.Atoms objects to transfer to the model
        noise : float
            Noise level
        jitter : float
            Jitter level
        
        """
        super().__init__(descriptor=descriptor, kernel=kernel, centralize=centralize,
                         **kwargs)
        
        self.jitter = jitter

        self.noise = noise

        self.transfer_data = []
        
        self.add_save_attributes(['Xn', 'Xm', 'K_mm', 'K_nm', 'Kmm_inv', 'L'])
        self.Xn = None
        self.Xm = None
        self.K_mm = None
        self.K_nm = None
        self.Kmm_inv = None
        self.L = None


    def _make_L(self, atoms_list, shape_X):
        """
        Make the L matrix

        Parameters
        ----------
        atoms_list : list of ase.Atoms
            List of ase.Atoms objects
        
        Returns
        -------
        np.ndarray
            L matrix
        
        """
        if len(atoms_list) == shape_X[0]:
            return np.eye(shape_X[0])
        
        lengths = [len(atoms) for atoms in atoms_list]
        r = len(lengths); c = np.sum(lengths)
        
        col = 0
        L = np.zeros((r,c))
        for i, atoms in enumerate(atoms_list):
            L[i,col:col+len(atoms)] = 1.
            col += len(atoms)
        return L

    
    def _update_L(self, new_atoms_list, shape_X):
        """
        Update the L matrix

        Parameters
        ----------
        new_atoms_list : list of ase.Atoms
            List of ase.Atoms objects

        Returns
        -------
        np.ndarray
            L matrix
        
        """
        if len(atoms_list) == shape_X[0]:
            new_size = shape_X[0] + self.L.shape[0]
            return np.eye(new_size)
        
        new_lengths = [len(atoms) for atoms in new_atoms_list]
        size = len(new_lengths)
        new_total_length = np.sum(new_lengths)
        new_L = np.zeros((self.L.shape[0]+size, self.L.shape[1]+new_total_length))
        new_L[0:self.L.shape[0], 0:self.L.shape[1]] = self.L

        for l in range(size):
            step = int(np.sum(new_lengths[:l]))
            new_L[l+self.L.shape[0], (self.L.shape[1]+step):(self.L.shape[1]+step+new_lengths[l])] = 1            
        return new_L
    
    @property
    def noise(self):
        """
        Noise level

        Returns
        -------
        float
            Noise level
        
        """
        return self._noise
    
    
    @noise.setter
    def noise(self, s):
        """
        Noise level

        Parameters
        ----------
        s : float
            Noise level
        
        """
        self._noise = s

        
    @property
    def transfer_data(self):
        """
        List of ase.Atoms objects to transfer to the model

        Returns
        -------
        list of ase.Atoms
            List of ase.Atoms objects to transfer to the model
        
        """
        return self._transfer_data

    
    @transfer_data.setter
    def transfer_data(self, l):
        """
        List of ase.Atoms objects to transfer to the model

        Parameters
        ----------
        l : list of ase.Atoms or dict of list with noise as key
            ase.Atoms objects to transfer to the model
        
        """
        if isinstance(l, list):
            self._transfer_data = l
            self._transfer_weights = np.ones(len(l))
        elif isinstance(l, dict):
            self._transfer_data = []
            self._transfer_weights = np.array([])
            for key, val in l.items():
                self._transfer_data += val
                self._transfer_weights = np.hstack((self._transfer_weights, float(key) * np.ones(len(val)) ))
        else:
            self._transfer_data = []
            self._trasfer_weights = np.array([])

    @property
    def transfer_weights(self):
        """
        Weights for the transfer data

        Returns
        -------
        np.ndarray
            Weights for the transfer data
        
        """
        return self._transfer_weights


    def _train_model(self):
        """
        Train the model

        """
        assert self.Xn is not None, 'self.Xn must be set prior to call'
        assert self.Xm is not None, 'self.Xm must be set prior to call'
        assert self.L is not None, 'self.L must be set prior to call'
        assert self.Y is not None, 'self.Y must be set prior to call'

        self.K_mm = self.kernel(self.Xm)
        self.K_nm = self.kernel(self.Xn, self.Xm)

        LK_nm = self.L @ self.K_nm
        K = self.K_mm + LK_nm.T @ self.sigma_inv @ LK_nm + self.jitter*np.eye(self.K_mm.shape[0])
        
        K = self.symmetrize(K)
        Q, R = qr(K)
        self.K_inv = lstsq(R, Q.T)[0]
        self.alpha = lstsq(R, Q.T @ LK_nm.T @ self.sigma_inv @ self.Y)[0]


    @candidate_list_comprehension        
    def predict_forces(self, atoms, **kwargs):
        """Method for forces prediction. 

        Parameters
        ----------
        atoms : ase.Atoms
            ase.Atoms object to predict forces for
        
        Returns
        ----------
        np.array 
            The force prediction with shape (N,3), where N is len(atoms)

        """        
        return self.predict_forces_central(atoms, **kwargs)

    
    @candidate_list_comprehension
    def predict_local_energy(self, atoms, **kwargs):
        """
        Calculate the local energies in the model.

        Parameters
        ----------
        atoms : ase.Atoms
            ase.Atoms object
        X : np.ndarray
            Features for the ase.Atoms object

        Returns
        -------
        np.ndarray
            Local energies
        
        """
        X = self.get_features(atoms)
        k = self.kernel(self.Xm, X)
        return (k.T@self.alpha).reshape(-1,) + self.single_atom_energies[atoms.get_atomic_numbers()]
    

    def _preprocess(self, data):
        """
        Preprocess the training data for the model

        Parameters
        ----------
        data : list of ase.Atoms
            List of ase.Atoms objects

        Returns
        -------
        np.ndarray
            Features for the ase.Atoms objects
        np.ndarray
            Energies for the ase.Atoms objects
        
        """
        X, Y = super()._preprocess(self.transfer_data + data)
        self.Xn = X

        self.L = self._make_L(self.transfer_data + data, X.shape)
        self.sigma_inv = self._make_sigma(self.transfer_data + data)
        self.Xm, _ = self.sparsifier(self.Xn)
        
        return self.Xm, Y


    def _update(self, data):
        """
        Update the training data for the model

        Parameters
        ----------
        data : list of ase.Atoms
            List of ase.Atoms objects

        Returns
        -------
        np.ndarray
            Features for the ase.Atoms objects
        np.ndarray
            Energies for the ase.Atoms objects
        
        """
        new, old = self._get_new_data(data)
        if len(new) == len(data):
            return self._preprocess(data)
        
        X_new, Y_new = super()._preprocess(new)

        X = np.vstack((self.X, X_new))
        self.Xn = X

        Y = np.vstack((self.Y, Y_new))

        self.L = self._update_L(new, X_new.shape)
        self.sigma_inv = self._make_sigma(self.transfer_data + data)
        self.Xm, _ = self.sparsifier(self.Xn)        

        return self.Xm, Y
        

    def _make_sigma(self, atoms_list):
        """
        Make the sigma matrix

        Parameters
        ----------
        atoms_list : list of ase.Atoms
            List of ase.Atoms objects

        Returns
        -------
        np.ndarray
            Sigma matrix
        
        """
        sigma_inv = np.diag([1/(len(atoms)*self.noise**2) for atoms in atoms_list])
        weights = np.ones(len(atoms_list))
        weights[:len(self.transfer_weights)] = self.transfer_weights
        sigma_inv[np.diag_indices_from(sigma_inv)] *= weights
        return sigma_inv


    def symmetrize(self, A):
        """
        Symmetrize a matrix

        Parameters
        ----------
        A : np.ndarray
            Matrix to symmetrize

        Returns
        -------
        np.ndarray
            Symmetrized matrix
        
        """
        return (A + A.T)/2
        

    
