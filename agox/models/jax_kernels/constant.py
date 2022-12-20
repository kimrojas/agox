from functools import partial
from typing import Optional, Union

import jax.numpy as np
from jax import jacfwd, jit

from bootcamp2022.gpr.kernels.ABC_kernel import KernelBaseClass
from bootcamp2022.helpers import cdist


class Constant(KernelBaseClass):
    """Kernel which returns a constant value.
    """

    hps = 1

    def __init__(self, constant: float = 1., constant_bounds : tuple = (1e-3, 1e5)):
        """Kernel which returns a constant value.

        Parameters
        ----------
        theta : float
            Constant value to return.
        """
        super().__init__()
        self.constant = constant
        self.constant_bounds = constant_bounds

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None, theta: Union[float, np.ndarray] = None) -> np.ndarray:
        if theta is None:
            c = self.constant
        else:
            c = np.exp(theta[0])

        if Y is None:
            Y = X

        return c * np.ones((len(X), len(Y)))

    def theta_gradient(self, X: np.ndarray, Y: Optional[np.ndarray] = None,
                       theta: Optional[np.ndarray] = None) -> np.ndarray:
        if theta is None:
            c = self.constant
        else:
            c = np.exp(theta[0])
        
        if Y is None:
            Y = X

        K = np.full((X.shape[0], X.shape[0]), c)
        return K, K[:,:,np.newaxis]

    def feature_gradient(self, X: np.ndarray, x: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        return 0
    
    def hyperparameters_to_array(self) -> np.ndarray:
        return np.array([self.constant])

    def hyperparameters_from_array(self, array):
        self.constant = array[0]

    def bounds_to_array(self) -> np.ndarray:
        return np.array([self.constant_bounds])

    def bounds_from_array(self, array):
        self.constant_bounds = array[0]
