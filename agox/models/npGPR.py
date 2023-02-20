import numpy as np
from functools import partial
from scipy.linalg import cho_solve, cho_factor
from scipy.optimize import fmin_l_bfgs_b

from agox.models.ABC_model import ModelBaseClass


class GlobalGPR(ModelBaseClass):
    name = 'ModelGPR'
    implemented_properties = ['energy', 'forces']

    def __init__(self, descriptor, kernel, prior=None, n_optimize=-1,
                 optimizer_maxiter=100, **kwargs):    
        ModelBaseClass.__init__(self, **kwargs)
        
        self.descriptor = descriptor
        self.kernel = kernel
        self.prior = prior
        self.n_optimize = n_optimize        
        self.optimizer_maxiter = optimizer_maxiter
        
    def predict_energy(self, atoms, **kwargs):
        if self.alpha is None:
            return self.postprocess_energy(atoms, 0)
        
        x = np.array(self.descriptor.get_global_features(atoms)).reshape(1,-1)
        k = self.kernel(self.X, x)
        e_pred = float(k.T @ self.alpha)
        return self.postprocess_energy(atoms, e_pred)

    def predict_uncertainty(self, atoms):
        if self.alpha is None:
            return 0
        
        x = np.array(self.descriptor.get_global_features(atoms)[0]).reshape(1,-1)
        k = self.kernel(self.X, x)
        k0 = self.kernel(x, x)
        var = float(k0 - k.T @ self.K_inv @ k)
        return np.sqrt(max(var, 0))

    def predict_forces(self, atoms):
        if self.alpha is None:
            return self.postprocess_forces(atoms, np.zeros((len(atoms), 3)))
        
        # F_i = - dE / dr_i = dE/dk dk/df df/dr_i = - alpha dk/df df_dr_i
        x = np.array(self.descriptor.get_global_features(atoms)[0]).reshape(1, -1)
        dfdr = np.array(self.descriptor.get_global_feature_gradient(atoms)[0])
        dkdf = self.kernel.get_feature_gradient(self.X, x)
        dkdr = np.dot(dkdf, dfdr.T)
        f_pred = -np.dot(dkdr.T, self.alpha).reshape(-1,3)
        return self.postprocess_forces(atoms, f_pred)

    def predict_forces_uncertainty(self, atoms):
        if self.alpha is None:
            return np.zeros((len(atoms), 3))
        
        x = np.array(self.descriptor.get_global_features(atoms)[0]).reshape(1, -1)

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
    
    def train_model(self, training_data, **kwargs):
        self.X, self.Y = self.preprocess(training_data)
        self.hyperparameter_search()
        
        self.K = self.kernel(self.X)
        self.alpha, self.K_inv, _ = self._solve(self.K, self.Y)
        self.ready_state = True
    
    def hyperparameter_search(self):
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
        L, lower = cho_factor(K)
        alpha = cho_solve((L, lower), Y)
        K_inv = cho_solve((L, lower), np.eye(K.shape[0]))
        return alpha, K_inv, (L, lower)

    def _marginal_log_likelihood(self, theta):
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

    
    def preprocess(self, data):
        Y = np.expand_dims(np.array([d.get_potential_energy() for d in data]), axis=1)
        
        if self.prior is None:
            self.prior_energy = np.zeros(Y.shape)            
        else:
            self.prior_energy = np.expand_dims(np.array([self.prior.predict_energy(d) for d in data]), axis=1)

        Y -= self.prior_energy
        self.mean_energy = np.mean(Y)
        Y -= self.mean_energy
        
        X = np.array(self.descriptor.get_global_features(data))
        return X, Y

    def postprocess_energy(self, atoms, e_pred):
        if self.prior is None:
            prior = 0
        else:
            prior = self.prior.predict_energy(atoms)
            
        return e_pred + prior + self.mean_energy

    def postprocess_forces(self, atoms, f_pred):
        if self.prior is None:
            prior = 0
        else:
            prior = self.prior.predict_forces(atoms)
            
        return f_pred + prior

    @classmethod
    def default(cls, environment=None, database=None, temp_atoms=None,
                lambda1min=1e-1, lambda1max=1e3, lambda2min=1e-1, lambda2max=1e3, 
                theta0min=1, theta0max=1e5, beta=0.01, use_delta_func=True, sigma_noise = 1e-2,
                descriptor=None, kernel=None, max_iterations=None, max_training_data=1000):

        from ase import Atoms
        from agox.models.priors.repulsive import Repulsive
        from agox.models.kernels import RBF, Constant as C, Noise
        from agox.models.gaussian_process.GPR import GPR

        if temp_atoms is None:
            temp_atoms = environment.get_template()
            temp_atoms += Atoms(environment.get_numbers())

        if descriptor is None:
            from agox.models.descriptors import Fingerprint
            descriptor = Fingerprint(temp_atoms)

        lambda1ini = (lambda1max - lambda1min)/2 + lambda1min
        lambda2ini = (lambda2max - lambda2min)/2 + lambda2min
        theta0ini = (theta0max - theta0min)/2 + theta0min
        
        if kernel is None:
            kernel = C(theta0ini, (theta0min, theta0max)) * \
            ( \
            C((1-beta), ((1-beta), (1-beta))) * RBF(lambda1ini, (lambda1min,lambda1max)) + \
            C(beta, (beta, beta)) * RBF(lambda2ini, (lambda2min,lambda2max)) 
            ) + \
            Noise(sigma_noise, (sigma_noise,sigma_noise))

        if use_delta_func:
            prior = Repulsive(rcut=6.)
        else:
            prior = None
        
        gpr = cls(kernel=kernel,
                  descriptor=descriptor,
                  prior=prior,
                  n_optimize=1)

        return gpr

        
