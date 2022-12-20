from functools import partial
from typing import Optional, Union

from abc import ABC, abstractmethod

import jax.numpy as np
from jax import jacfwd, jit


class KernelBaseClass(ABC):
    

    def __init__(self):
        """ Base class for all kernels.
        """

        
    @abstractmethod
    def hyperparameters_to_array(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def hyperparameters_from_array(self, array):
        raise NotImplementedError()


    @abstractmethod
    def bounds_to_array(self) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def bounds_from_array(self, array):
        raise NotImplementedError()    

    @abstractmethod
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate the kernel k(X, Y).

        Parameters
        ----------
        X : np.ndarray
            Left argument of the returned kernel.
        Y : np.ndarray, optional
            Right argument of the returned kernel. If None, Y = X.
        theta : np.ndarray, optional
            Optionally override the kernel's hyperparameters.

        Returns
        -------
        np.ndarray
            The evaluated kernel.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def hps(self) -> int:
        raise NotImplementedError()        

    @property
    def theta(self):
        """ The log-transformed hyperparameters as array

        """        
        return np.log(self.hyperparameters_to_array())

    @theta.setter
    def theta(self, theta):
        self.hyperparameters_from_array(np.exp(theta))


    @property
    def bounds(self):
        """ The log-transformed hyperparameters bounds as array

        """        
        return np.log(self.bounds_to_array())

    @bounds.setter
    def bounds(self, array):
        if len(array) != self.hps:
            raise ValueError('Theta does not have the correct number of entries.')

        self._bounds = np.exp(array)
        

    @partial(jit, static_argnums=(0,))
    def feature_gradient(self, X: np.ndarray, x: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate the gradient of the kernel with respect to a single feature
        vector.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        x : np.ndarray
            Feature vector.
        theta: np.ndarray, optional
            Optionally override the kernel's hyperparameter(s).

        Returns
        -------
        np.ndarray
            Gradient of the kernel with respect to the feature vector.
        """
        grad = jacfwd(lambda _x: self(X, _x ,theta))
        return np.squeeze(grad(x))


    @partial(jit, static_argnums=(0,))
    def theta_gradient(self, X: np.ndarray, Y: Optional[np.ndarray] = None, theta: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate the gradient of the kernel with respect to the
        hyperparameters theta.

        Parameters
        ----------
        X : np.ndarray
            Left argument of the kernel.
        Y : np.ndarray, optional
            Right argument of the kernel. If None, Y = X.
        theta : np.ndarray, optional
            Optionally override the kernel's hyperparameters.

        Returns
        -------
        np.ndarray
            Gradient of the kernel with respect to the hyperparameter(s).
        """
        if theta is None:
            theta = self.theta

        f = lambda t: self(X,Y,t)
        grad = jacfwd(f)
        gradient = grad(theta)

        if np.ndim(gradient) == 2:
            gradient = gradient[:, :, np.newaxis]
            
        return gradient

    def __add__(self, k):
        from bootcamp2022.gpr.kernels.constant import Constant

        if not isinstance(k, KernelBaseClass):
            return SumKernel(self, Constant(k))

        return SumKernel(self, k)

    def __radd__(self, k):
        from bootcamp2022.gpr.kernels.constant import Constant
        
        if not isinstance(k, KernelBaseClass):
            return SumKernel(Constant(k), self)

        return SumKernel(k, self)    

    def __mul__(self, k):
        from bootcamp2022.gpr.kernels.constant import Constant
                
        if not isinstance(k, KernelBaseClass):
            return ProductKernel(self, Constant(k))

        return ProductKernel(self, k)
    
    def __rmul__(self, k):
        from bootcamp2022.gpr.kernels.constant import Constant
                
        if not isinstance(k, KernelBaseClass):
            return ProductKernel(Constant(k), self)

        return ProductKernel(k, self)
    
    def __pow__(self, k: float):
            return ExponentKernel(self, k)
    

    # def __repr__(self) -> str:
    #     if isinstance(self.theta, np.ndarray):
    #         params = f'({", ".join(self.theta)})'
    #     elif isinstance(self.theta, (Number, np.number)):
    #         params = f'({self.theta})'
    #     else:
    #         params = ''
    #     return f'{self.__class__.__name__}{params}'




class KernelOperator(KernelBaseClass):
    """Base class for all kernel operators.
    """

    def __init__(self, k1: KernelBaseClass, k2: KernelBaseClass):
        """
        Parameters
        ----------
        k1 : KernelBaseClass
            First kernel to apply the operator to.
        k2 : KernelBaseClass
            Second kernel to apply the operator to.
        """

        self.k1 = k1
        self.k2 = k2

    @property
    def hps(self) -> int:
        return self.k1.hps + self.k2.hps
    
    def hyperparameters_to_array(self) -> np.ndarray:
        return np.hstack((self.k1.hyperparameters_to_array(),
                          self.k2.hyperparameters_to_array()))

    def hyperparameters_from_array(self, array):
        self.k1.hyperparameters_from_array(array[:self.k1.hps])
        self.k2.hyperparameters_from_array(array[self.k1.hps:])


    def bounds_to_array(self) -> np.ndarray:
        return np.vstack((self.k1.bounds_to_array(),
                          self.k2.bounds_to_array()))


    def bounds_from_array(self, array):
        self.k1.bounds = array[:self.k1.hps]
        self.k2.bounds = array[self.k1.hps:]

    

class SumKernel(KernelOperator):
    """Combine two kernels via the sum operator.
    """

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None, theta: np.ndarray = None) -> np.ndarray:
        if theta is None:
            return self.k1(X, Y) + self.k2(X, Y)
        else:
            return self.k1(X, Y, theta[:self.k1.hps]) + self.k2(X, Y, theta[self.k1.hps:])

    def feature_gradient(self, X: np.ndarray, x: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        if theta is None:
            return self.k1.feature_gradient(X, x) + self.k2.feature_gradient(X, x)
        else:
            return self.k1.feature_gradient(X, x, theta[:self.k1.hps]) + self.k2.feature_gradient(X, x, theta[self.k1.hps:])

    def theta_gradient(self, X: np.ndarray, Y: Optional[np.ndarray] = None,
                       theta: Optional[np.ndarray] = None) -> np.ndarray:
        if theta is None:
            return np.dstack((self.k1.theta_gradient(X, Y), self.k2.theta_gradient(X, Y)))
        else:
            return np.dstack((self.k1.theta_gradient(X, Y, theta[:self.k1.hps]),
                              self.k2.theta_gradient(X, Y, theta[self.k1.hps:])))

        


class ProductKernel(KernelOperator):
    """Combine two kernels via the product operator.
    """

    @partial(jit, static_argnums=(0,))    
    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None, theta: np.ndarray = None) -> np.ndarray:
        if theta is None:
            return self.k1(X, Y) * self.k2(X, Y)
        else:
            return self.k1(X, Y, theta[:self.k1.hps]) * self.k2(X, Y, theta[self.k1.hps:])        


    def feature_gradient(self, X: np.ndarray, x: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        if theta is None:
            return self.k1(X, x)*self.k2.feature_gradient(X, x) + self.k1.feature_gradient(X, x)*self.k2(X, x)
        else:
            return self.k1(X, x, theta[:self.k1.hps])*self.k2.feature_gradient(X, x, theta[self.k1.hps:]) + \
                self.k1.feature_gradient(X, x, theta[:self.k1.hps])*self.k2(X, x, theta[self.k1.hps:])
        

    @partial(jit, static_argnums=(0,))    
    def theta_gradient(self, X: np.ndarray, Y: Optional[np.ndarray] = None,
                       theta: Optional[np.ndarray] = None) -> np.ndarray:
        if theta is None:
             #K1 = self.k1(X, Y)
            K1, K1_grad = self.k1.theta_gradient(X, Y)
            # K2 = self.k2(X, Y)
            K2, K2_grad = self.k2.theta_gradient(X, Y)
        else:
            # K1 = self.k1(X, Y, theta[:self.k1.hps])
            K1, K1_grad = self.k1.theta_gradient(X, Y, theta[:self.k1.hps])
            # K2 = self.k2(X, Y, theta[self.k1.hps:])
            K2, K2_grad = self.k2.theta_gradient(X, Y, theta[self.k1.hps:])

        return K1*K2, np.dstack((K1_grad*K2[:,:,np.newaxis], K2_grad*K1[:,:,np.newaxis] ))
        

class Exponent(KernelBaseClass):
    """Combine a kernel and a constant via the exponent operator.
    """

    def __init__(self, k: KernelBaseClass, exp: float):
        """
        Parameters
        ----------
        k : KernelBaseClass
            Base kernel to apply the operator to.
        exp : float
            Exponent for the base kernel.
        """
        self.k = k
        self.exp = exp

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.k(X, Y) ** self.exp

    @property
    def hps(self) -> int:
        return self.k.hps + 1
    
    def hyperparameters_to_array(self) -> np.ndarray:
        return np.hstack((self.k.hyperparameters_to_array(), np.array([self.exp])))

    def hyperparameters_from_array(self, array):
        self.k.theta = array[:self.k.hps]

    
