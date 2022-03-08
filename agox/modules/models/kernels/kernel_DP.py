import awkward as ak
from numba import njit
import numpy as np
from agox.modules.models.kernels.kernel_ABC import KernelBaseClass

@njit
def _get_global_kernel(features_N, features_M, symmetric, zeta):
    N = len(features_N)
    M = len(features_M)
    K = np.zeros((N, M))
    
    if not symmetric:
        for i in range(N):
            for j in range(M):
                K[i][j] = _global_covariance_fast(features_N[i], features_M[j], zeta)
        
    elif symmetric:
        for i in range(N):
            for j in range(i, N):
                K[i][j] = _global_covariance_fast(features_N[i], features_M[j], zeta)
                if i != j:
                    K[j][i] = K[i][j]
    return K


@njit
def _get_global_kernel_derivative(features, zeta):
    N = len(features)
    K = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i, N):
            K[i][j] = _global_covariance_derivative(features[i], features[j], zeta)
            if i != j:
                K[j][i] = K[i][j]
    return K


@njit
def _global_covariance(X, Y, zeta):
    N_X = len(X)
    N_Y = len(Y)
    k = 0
    for ix in range(N_X):
        for iy in range(N_Y):
            d = 0
            n0 = 0
            n1 = 0
            for i in range(len(X[ix])):
                d += X[ix][i] * Y[iy][i]
                n0 += X[ix][i] * X[ix][i]
                n1 += Y[iy][i] * Y[iy][i]
            k += np.power(d/np.sqrt(n0*n1), zeta)
    return k

@njit
def _global_covariance_fast(X, Y, zeta):
    N_X = len(X)
    N_Y = len(Y)

    tmp_X2 = np.empty(N_X)
    for ix in range(N_X):
        s = 0
        for j in range(len(X[ix])):
            s += X[ix][j]*X[ix][j]
        tmp_X2[ix] = np.sqrt(s)

    tmp_Y2 = np.empty(N_Y)
    for iy in range(N_Y):
        s = 0
        for j in range(len(Y[iy])):
            s += Y[iy][j]*Y[iy][j]
        tmp_Y2[iy] = np.sqrt(s)

    prod = np.dot(X, Y.T)

    k = 0
    for ix in range(N_X):
        for iy in range(N_Y):
            k += np.power(prod[ix][iy]/(tmp_X2[ix]*tmp_Y2[iy]), zeta)
    return k



@njit
def _global_covariance_derivative(X, Y, zeta):
    N_X = len(X)
    N_Y = len(Y)
    k = 0
    for ix in range(N_X):
        for iy in range(N_Y):
            d = 0
            n0 = 0
            n1 = 0
            for i in range(len(X[ix])):
                d += X[ix][i] * Y[iy][i]
                n0 += X[ix][i] * X[ix][i]
                n1 += Y[iy][i] * Y[iy][i]
            k += zeta*np.power(d/np.sqrt(n0*n1), zeta-1)
    return k


@njit
def _local_kernel(X, Y, zeta):
    N_X = len(X)
    N_Y = len(Y)    
    k = np.zeros((N_X, N_Y))
    for ix in range(N_X):
        for iy in range(N_Y):
            d = 0
            n0 = 0
            n1 = 0
            for i in range(len(X[ix])):
                d += X[ix][i] * Y[iy][i]
                n0 += X[ix][i] * X[iy][i]
                n1 += Y[ix][i] * Y[iy][i]
            k[ix][iy] = np.power(d/np.sqrt(n0*n1), zeta)
    return k



class DPKernel(KernelBaseClass):
    name = 'DBKernel'
    
    def __init__(self, zeta=10, zeta_bounds=(1,1000)):
        self._zeta = zeta
        self.hyperparameter_bounds = [zeta_bounds]
        self.hyperparameters = np.array([zeta])

    def _set_hyperparameters(self, hyperparameters):
        self._zeta = int(hyperparameters[0])

    @property
    def zeta(self):
        return self._zeta
    @zeta.setter
    def zeta(self, val):
        self._hyperparameters = np.array([val])
        self._zeta = val


    def _get_global_kernel(self, features_N, features_M, symmetric=False):
        return _get_global_kernel(features_N, features_M, symmetric, self.zeta)

    def get_kernel_gradient(self, features):
        if len(features.shape) == 3: # non-jagged!
            jac = _get_global_kernel_derivative(features, self.zeta)
        else: # jagged - make awkward
            jagged_features = ak.Array([a for a in features])
            jac = _get_global_kernel_derivative(jagged_features, self.zeta)
        jac = np.expand_dims(jac, axis=0)
        return jac

    def get_local_kernel(self, features_N, features_M):
        return _local_kernel(features_N, features_M, self.zeta)
