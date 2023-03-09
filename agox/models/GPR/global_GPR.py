import numpy as np
import warnings
from scipy.linalg import cho_solve, cho_factor
from scipy.optimize import fmin_l_bfgs_b

from ase.calculators.calculator import all_changes
from agox.models.GPR.ABC_GPR import GPRBaseClass


class GlobalGPR(GPRBaseClass):
    
    name = 'GlobalGPR'

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
                 n_optimize=1, optimizer_maxiter=100, **kwargs):
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


    def get_features(self, atoms):
        """
        Get features for a given atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        np.ndarray
            Features
        
        """
        f = self.descriptor.get_global_features(atoms)

        if isinstance(f, np.ndarray) and len(f.shape) == 1:
            f = f.reshape(1, -1)
        f = np.vstack(f)
        
        return f
        
        

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
        x = self.get_features(atoms)
        dfdr = np.array(self.descriptor.get_global_feature_gradient(atoms))
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
        
        x = self.get_features(atoms)

        dfdr = np.array(self.descriptor.get_global_feature_gradient(atoms))
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


    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        if 'uncertainty' in properties:
            self.results['uncertainty'] = self.predict_uncertainty(atoms)
        if 'force_uncertainty' in properties:
            self.results['force_uncertainty'] = self.predict_forces_uncertainty(atoms)

    
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
    


    def get_model_parameters(self):
        warnings.warn('get_model_parameters is deprecated and will be removed in a future release.', DeprecationWarning)        
        parameters = {}
        parameters['feature_mat'] = self.X
        parameters['alpha'] = self.alpha
        parameters['bias'] = self.mean_energy
        parameters['kernel_hyperparameters'] = self.kernel.get_params()
        parameters['K_inv'] = self.K_inv
        parameters['iteration'] = self.get_iteration_counter()
        return parameters
    
    def set_model_parameters(self, parameters):
        warnings.warn('set_model_parameters is deprecated and will be removed in a future release.', DeprecationWarning)
        self.X = parameters['feature_mat']
        self.alpha = parameters['alpha']
        self.mean_energy = parameters['bias']
        self.K_inv = parameters['K_inv']
        self.set_iteration_counter(parameters['iteration'])
        self.kernel.set_params(**parameters['kernel_hyperparameters'])
        self.ready_state = True


    def get_feature_calculator(self):
        warnings.warn("The 'get_feature_calculator'-method will be deprecated in a future release.", DeprecationWarning)
        return self.descriptor

        
    @classmethod
    def default(cls, environment=None, database=None, temp_atoms=None, lambda1min=1e-1, lambda1max=1e3, lambda2min=1e-1, lambda2max=1e3, 
                theta0min=1, theta0max=1e5, beta=0.01, use_delta_func=True, sigma_noise=1e-2,
                descriptor=None, kernel=None, max_iterations=9999, max_training_data=1000):

        """
        Creates a GPR model. 

        Parameters
        ------------
        environment: AGOX environment. 
            Used to create an atoms object to initialize e.g. the feature calculator. 
        database: 
            AGOX database that the model will be attached to. 
        lambda1min/lambda1max: float
            Length scale minimum and maximum. 
        lambda2min/lambda2max: float
            Length scale minimum and maximum. 
        theta0min/theta0max: float
            Amplitude minimum and maximum 
        use_delta_func: bool
            Whether to use the repulsive prior function. 
        sigma_noise: float
            Noise amplitude. 
        feature_calculator: object
            A feature calculator object, if None defaults to reasonable 
            fingerprint settings. 
        kernel: str or kernel object or None. 
            If kernel='anisotropic' the anisotropic RBF kernel is used where
            radial and angular componenets are treated at different length scales
            If None the standard RBF is used. 
            If a kernel object then that kernel is used. 
        max_iterations: int or None
            Maximum number of iterations for the hyperparameter optimization during 
            its BFGS optimization through scipy. 
        """
        
        from ase import Atoms
        from agox.models.priors.repulsive import Repulsive
        from agox.models.GPR.kernels import RBF, Constant as C, Noise

        assert temp_atoms is not None or environment is not None

        if temp_atoms is None:
            temp_atoms = environment.get_template()
            temp_atoms += Atoms(environment.get_numbers())

        if descriptor is None:
            from agox.models.descriptors import Fingerprint
            descriptor = Fingerprint(temp_atoms, use_cache=True)

        lambda1ini = (lambda1max - lambda1min)/2 + lambda1min
        lambda2ini = (lambda2max - lambda2min)/2 + lambda2min
        theta0ini = 5000                         
        
        if kernel is None:
            kernel = C(theta0ini, (theta0min, theta0max)) * \
            ( \
            C((1-beta), ((1-beta), (1-beta))) * RBF(lambda1ini, (lambda1min,lambda1max)) + \
            C(beta, (beta, beta)) * RBF(lambda2ini, (lambda2min,lambda2max)) 
            ) + \
            Noise(sigma_noise, (sigma_noise,sigma_noise))
            

        if use_delta_func:
            delta = Repulsive(rcut=6)
        else:
            delta = None
        
        return cls(database=database, kernel=kernel, descriptor=descriptor, prior=delta,
                   n_optimize=1, optimizer_maxiter=max_iterations)


    
        
