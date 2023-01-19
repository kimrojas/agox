import jax.numpy as np
import numpy
from functools import partial
from jax import jit, vmap, grad, random
from jax.scipy.linalg import cho_solve, cho_factor
from scipy.optimize import fmin_l_bfgs_b

from bootcamp2022.gpr.ABC_model import ModelBaseClass

from time import time, sleep


class GPR(ModelBaseClass):
    name = 'JaxGPR'
    implemented_properties = ['energy', 'forces']

    def __init__(self, descriptor, kernel, noise=0.1, n_optimize=-1, seed=0, optimizer_maxiter=100,**kwargs):    
        ModelBaseClass.__init__(self, **kwargs)
        
        self.descriptor = descriptor
        self.kernel = kernel
        self.noise = noise
        self.n_optimize = n_optimize        
        self.key = random.PRNGKey(seed)
        self.optimizer_maxiter = optimizer_maxiter
        
    def predict_energy(self, atoms):
        x = np.array(self.descriptor.get_global_features(atoms)[0]).reshape(1,-1)
        k = self.kernel(x, self.X)
        e_pred = float(k @ self.alpha)
        return self.postprocess(e_pred)

    def predict_forces(self, atoms):
        # F_i = - dE / dr_i = dE/dk dk/df df/dr_i = - alpha dk/df df_dr_i
        f = np.array(self.descriptor.get_global_features(atoms)[0]).reshape(1, -1)
        dfdr = np.array(self.descriptor.get_global_feature_gradient(atoms)[0])
        dkdf = self.kernel.get_feature_gradient(self.X, f)
        dkdr = np.dot(dkdf, dfdr.T)
        return - np.dot(dkdr.T, self.alpha).reshape(-1,3)
    
    def train_model(self, training_data):
        self.X, self.Y = self.preprocess(training_data)
        # self.K = self.kernel(self.X)
        self.hyperparameter_search()
        self.K = self.kernel(self.X)
        self.alpha, _ = self._solve_alpha(self.K, self.Y)
    
    def hyperparameter_search(self):
        initial_parameters = []
        initial_parameters.append(self.kernel.theta.copy())
        if self.n_optimize > 0:
            for _ in range(self.n_optimize):
                self.key, key = random.split(self.key)
                init_theta = random.uniform(key, shape=(len(self.kernel.bounds),), minval=self.kernel.bounds[:,0], maxval=self.kernel.bounds[:,1])
                initial_parameters.append(init_theta)
            
            fmins = []
            thetas = []
            for init_theta in initial_parameters:
                theta_min, nll_min = self.hyperparameter_optimize(init_theta=init_theta)
                fmins.append(nll_min)
                thetas.append(theta_min)

            self.kernel.theta = thetas[np.argmin(np.array(fmins))]


    # @partial(jit, static_argnums=(0,))    
    def hyperparameter_optimize(self, init_theta=None):
        def f(theta):
            P, grad_P = self._marginal_log_likelihood_gradient(theta)
            if np.isnan(P):
                return np.inf, numpy.zeros_like(theta, dtype='float64')
            P, grad_P = -float(P), -numpy.asarray(grad_P, dtype='float64')
            return P, grad_P

        bounds = self.kernel.bounds
        
        if init_theta is None:
            self.key, key = random.split(self.key)
            init_theta = random.uniform(key, shape=(len(bounds),), minval=bounds[:,0], maxval=bounds[:,1])
            
        theta_min, fmin, conv = fmin_l_bfgs_b(f, numpy.asarray(init_theta, dtype='float64'),
                                              bounds=numpy.asarray(bounds, dtype='float64'),
                                              maxiter=self.optimizer_maxiter)

        return theta_min, fmin
    
    @partial(jit, static_argnums=(0,))
    def _solve_alpha(self, K, Y):
        L, lower = cho_factor(K)
        alpha = cho_solve((L, lower), Y)
        return alpha, (L, lower)

    @partial(jit, static_argnums=(0,))    
    def _solve(self, K, Y):
        L, lower = cho_factor(K)
        alpha = cho_solve((L, lower), Y)
        K_inv = cho_solve((L, lower), np.eye(K.shape[0]))
        return alpha, K_inv, (L, lower)

    # @partial(jit, static_argnums=(0,))
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

    # @partial(jit, static_argnums=(0,))
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
        self.mean_energy = np.mean(Y)
        Y -= self.mean_energy
        X = np.array(self.descriptor.get_global_features(data))
        return X, Y

    def postprocess(self, e_pred):
        return e_pred + self.mean_energy            
