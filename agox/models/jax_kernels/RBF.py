from functools import partial
from typing import Optional, Union

import jax.numpy as np
from bootcamp2022.helpers import cdist
from bootcamp2022.gpr.kernels.ABC_kernel import KernelBaseClass
from jax import jacfwd, jit


class RBF(KernelBaseClass):
    """Kernel which returns the radial basis function.
    """

    hps = 1
    
    def __init__(self, length_scale: float = 1., length_scale_bounds: tuple = (1e-5, 1e4)):
        """The isotropic radial basis function kernel.

        Parameters
        ----------
        theta : float
            Length scale for the kernel.
        """
        super().__init__()
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds


    @partial(jit, static_argnums=(0,))        
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None,
                 theta: Optional[Union[float, np.ndarray]] = None) -> np.ndarray:
        if theta is None:
            ls = self.length_scale
        else:
            ls = np.exp(theta[0])
            
        if Y is None:
            Y = X

        dists = cdist(X / ls, Y / ls)
        K = np.exp(-0.5 * dists)
        
        return K

    def hyperparameters_to_array(self) -> np.ndarray:
        return np.array([self.length_scale])

    def hyperparameters_from_array(self, array):
        self.length_scale = array[0]


    def bounds_to_array(self) -> np.ndarray:
        return np.array([self.length_scale_bounds])

    def bounds_from_array(self, array):
        self.length_scale_bounds = array[0]

    
    @partial(jit, static_argnums=(0,))
    def feature_gradient(self, X: np.ndarray, x: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        if theta is None:
            ls = self.length_scale
            theta = self.theta
        else:
            ls = np.exp(theta[0])

        if np.ndim(x) == 1:
            x = x[np.newaxis, :]
            
        K = self(X, x, theta)
        dK_dd = -1 / (2 * ls ** 2) * K
        dd_df = -2 * (X - x)
        dk_df = np.multiply(dK_dd, dd_df)
        
        return dk_df

    @partial(jit, static_argnums=(0,))
    def theta_gradient(self, X: np.ndarray, Y: Optional[np.ndarray] = None,
                       theta: Optional[np.ndarray] = None) -> np.ndarray:

        if theta is None:
            ls = self.length_scale
        else:
            ls = np.exp(theta[0])
            
        if Y is None:
            Y = X
        
        dists = cdist(X / ls, Y / ls)
        K = np.exp(-0.5 * dists)
        gradient = K * dists
        
        return K, gradient[:,:,np.newaxis]


    
