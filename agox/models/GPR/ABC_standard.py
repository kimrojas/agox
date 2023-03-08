from abc import abstractmethod
from agox.models.GPR.ABC_GPR import GPRBaseClass

import numpy as np
from functools import partial
from scipy.linalg import cho_solve, cho_factor
from scipy.optimize import fmin_l_bfgs_b


class StandardBaseClass(GPRBaseClass):

    implemented_properties = ['energy', 'forces', 'uncertainty', 'force_uncertainty']

    dynamic_attributes = ['alpha', 'K_inv', 'X', 'K']
    
    """
    Standard GPR Base Class

    Attributes
    ----------
    kernel : Kernel
        Kernel object
    descriptor : Descriptor
        Descriptor object
    centralize : bool
        Whether to centralize the data
    n_optimize : int
        Number of hyperparameter optimization runs
    optimizer_maxiter : int
        Maximum number of iterations for the optimizer
    X : np.ndarray
        Training features
    Y : np.ndarray
        Training targets
    K : np.ndarray
        Kernel matrix
    K_inv : np.ndarray
        Inverse of the kernel matrix
    alpha : np.ndarray
        Alpha vector

    Methods
    -------
        
    """
    def __init__(self, descriptor, kernel, centralize=True,
                 n_optimize=-1, optimizer_maxiter=100, **kwargs):
        """
        Parameters
        ----------
        descriptor : Descriptor
            Descriptor object
        kernel : Kernel
            Kernel object
        centralize : bool
            Whether to centralize the data
        n_optimize : int
            Number of hyperparameter optimization runs
        optimizer_maxiter : int
            Maximum number of iterations for the optimizer
        
        """
        super().__init__(descriptor, kernel, centralize=centralize,
                         **kwargs)
        self.n_optimize = n_optimize
        self.optimizer_maxiter = optimizer_maxiter

        
        

    def predict_uncertainty(self, atoms):
        """
        Predict uncertainty for a given atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        float
            Uncertainty
        
        """
        if self.alpha is None:
            return 0
        
        x = self.get_features(atoms).reshape(1,-1)
        k = self.kernel(self.X, x)
        k0 = self.kernel(x, x)
        var = float(k0 - k.T @ self.K_inv @ k)
        return np.sqrt(max(var, 0))

    
    def predict_forces(self, atoms):
        """
        Predict forces for a given atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        np.ndarray
            Forces
        """
        if self.alpha is None:
            return self.postprocess_forces(atoms, np.zeros((len(atoms), 3)))
        
        # F_i = - dE / dr_i = dE/dk dk/df df/dr_i = - alpha dk/df df_dr_i
        x = self.get_features(atoms).reshape(1, -1)
        dfdr = np.array(self.descriptor.get_global_feature_gradient(atoms)[0])
        dkdf = self.kernel.get_feature_gradient(self.X, x)
        dkdr = np.dot(dkdf, dfdr.T)
        f_pred = -np.dot(dkdr.T, self.alpha).reshape(-1,3)
        return self.postprocess_forces(atoms, f_pred)

    
    def predict_forces_uncertainty(self, atoms):
        """
        Predict forces uncertainty for a given atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        np.ndarray
            Forces uncertainty
        
        """
        if self.alpha is None:
            return np.zeros((len(atoms), 3))
        
        x = self.get_features(atoms).reshape(1, -1)

        dfdr = np.array(self.descriptor.get_global_feature_gradient(atoms)[0])
        dkdf = self.kernel.get_feature_gradient(self.X, x)
        dkdr = np.dot(dkdf, dfdr.T)
        
        k = self.kernel(self.X, x)
        k0 = self.kernel(x, x)
        var = k0 - k.T @ self.K_inv @ k
        if var < 0:
            return np.zeros((len(atoms), 3))
        else:
            return 1/np.sqrt(var) * dkdr.T @ self.K_inv @ k


    def _train_model(self):
        """
        Train the model
        
        """
        self.hyperparameter_search()
        self.K = self.kernel(self.X)
        self.alpha, self.K_inv, _ = self._solve(self.K, self.Y)


    def hyperparameter_search(self):
        """
        Hyperparameter search
        
        """
        initial_parameters = []
        initial_parameters.append(self.kernel.theta.copy())
        if self.n_optimize > 0:
            for _ in range(self.n_optimize-1):
                init_theta = np.random.uniform(size=(len(self.kernel.bounds),), low=self.kernel.bounds[:,0], high=self.kernel.bounds[:,1])
                initial_parameters.append(init_theta)

            fmins = []
            thetas = []
            for init_theta in initial_parameters:
                theta_min, nll_min = self._hyperparameter_optimize(init_theta=init_theta)
                fmins.append(nll_min)
                thetas.append(theta_min)

            self.kernel.theta = thetas[np.argmin(np.array(fmins))]


    def _hyperparameter_optimize(self, init_theta=None):
        """
        Hyperparameter optimization

        Parameters
        ----------
        init_theta : np.ndarray
            Initial theta

        Returns
        -------
        np.ndarray
            Optimal theta
        
        """
        def f(theta):
            P, grad_P = self._marginal_log_likelihood_gradient(theta)
            if np.isnan(P):
                return np.inf, np.zeros_like(theta, dtype='float64')
            P, grad_P = -float(P), -np.asarray(grad_P, dtype='float64')
            return P, grad_P

        bounds = self.kernel.bounds
        
        if init_theta is None:
            self.key, key = random.split(self.key)
            init_theta = random.uniform(key, shape=(len(bounds),), minval=bounds[:,0], maxval=bounds[:,1])
            
        theta_min, fmin, conv = fmin_l_bfgs_b(f, np.asarray(init_theta, dtype='float64'),
                                              bounds=np.asarray(bounds, dtype='float64'),
                                              maxiter=self.optimizer_maxiter)

        return theta_min, fmin
    

    def _solve(self, K, Y):
        """
        Solve the linear system

        Parameters
        ----------
        K : np.ndarray
            Kernel matrix
        Y : np.ndarray
            Target values

        Returns
        -------
        np.ndarray
            Alpha
        np.ndarray
            Inverse of kernel matrix
        np.ndarray
            Cholesky decomposition of kernel matrix
        
        """
        L, lower = cho_factor(K)
        alpha = cho_solve((L, lower), Y)
        K_inv = cho_solve((L, lower), np.eye(K.shape[0]))
        return alpha, K_inv, (L, lower)

    
    def _marginal_log_likelihood(self, theta):
        """
        Marginal log likelihood

        Parameters
        ----------
        theta : np.ndarray
            Kernel parameters

        Returns
        -------
        float
            Marginal log likelihood
        
        """
        t = self.kernel.theta.copy()
        self.kernel.theta = theta        
        K = self.kernel(self.X)
        self.kernel.theta = t        
        
        alpha, K_inv, (L, lower) = self._solve(K, self.Y)

        log_P = - 0.5 * np.einsum("ik,ik->k", self.Y, alpha) \
            - np.sum(np.log(np.diag(L))) \
            - K.shape[0] / 2 * np.log(2 * np.pi)

        return np.sum(log_P)

    
    def _marginal_log_likelihood_gradient(self, theta):
        """
        Marginal log likelihood gradient

        Parameters
        ----------
        theta : np.ndarray
            Kernel parameters

        Returns
        -------
        float
            Marginal log likelihood
        np.ndarray
            Marginal log likelihood gradient

        """
        t = self.kernel.theta.copy()
        self.kernel.theta = theta
        K, K_hp_gradient = self.kernel(self.X, eval_gradient=True)
        self.kernel.theta = t
        
        alpha, K_inv, (L, lower) = self._solve(K, self.Y)

        log_P = - 0.5 * np.einsum("ik,ik->k", self.Y, alpha) \
            - np.sum(np.log(np.diag(L))) \
            - K.shape[0] / 2 * np.log(2 * np.pi)
        
        inner = np.squeeze(np.einsum("ik,jk->ijk", alpha, alpha), axis=2) - K_inv
        inner = inner[:,:,np.newaxis]
        
        grad_log_P = np.sum(0.5 * np.einsum("ijl,ijk->kl", inner, K_hp_gradient), axis=-1)
        return log_P, grad_log_P
    
    
        
