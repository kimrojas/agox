from abc import abstractmethod
from agox.models.GPR.ABC_GPR import GPRBaseClass
import warnings
import numpy as np
from scipy.linalg import cholesky, cho_solve, qr, lstsq, LinAlgError
from time import time

from agox.writer import agox_writer
from agox.observer import Observer



class SparseBaseClass(GPRBaseClass):

    implemented_properties = ['energy', 'forces', 'local_energy']

    dynamic_attributes = ['Xm', 'K_inv', 'Kmm_inv', 'alpha']


    """
    Sparse GPR Base Class

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
    sparsify(X, atoms_list)
        Sparsify the training data
    
    """
    

    def __init__(self, descriptor, kernel, transfer_data=[], noise=0.05,
                 jitter=1e-8, **kwargs):

        """
        Sparse GPR Base Class

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
        super().__init__(descriptor, kernel, **kwargs)
        self.jitter = jitter
        self.transfer_data = transfer_data
        self.noise = noise

        self.Xn = None
        self.Xm = None

        self.K_mm = None
        self.K_nm = None
        self.K_inv = None
        self.Kmm_inv = None
        self.L = None


    @abstractmethod
    def _make_L(self, data):
        pass

    @abstractmethod
    def _update_L(self, data):
        pass

        
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

    

    def predict_local_energy(self, atoms=None, X=None):
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
        if X is None:
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

        self.L = self._make_L(self.transfer_data + data)
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

        self.L = self._update_L(new)
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
        

    def get_model_parameters(self):
        warnings.warn('get_model_parameters is deprecated and will be removed soon.', DeprecationWarning)
        parameters = {}
        parameters['Xm'] = self.Xm
        parameters['K_inv'] = self.K_inv
        parameters['Kmm_inv'] = self.Kmm_inv
        parameters['alpha'] = self.alpha
        parameters['single_atom_energies'] = self.single_atom_energies
        parameters['theta'] = self.kernel.theta
        return parameters

    def set_model_parameters(self, parameters):
        warnings.warn('set_model_parameters is deprecated and will be removed soon.', DeprecationWarning)
        self.Xm = parameters['Xm']
        self.X = parameters['Xm']
        self.K_inv = parameters['K_inv']
        self.Kmm_inv = parameters['Kmm_inv']
        self.alpha = parameters['alpha']
        self.single_atom_energies = parameters['single_atom_energies']
        self.kernel.theta = parameters['theta']
        self.ready_state = True

    
