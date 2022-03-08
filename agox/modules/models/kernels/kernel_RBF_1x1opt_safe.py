import awkward as ak
from numba import njit, prange
import numpy as np
from agox.modules.models.kernels.kernel_ABC import KernelBaseClass
from scipy.spatial.distance import cdist

def _get_global_kernel_1x1_opt(X, Y, symmetric, A, ls):
    l = ls*ls
    distances_sq = cdist(X, Y, metric='sqeuclidean')
    K = A*A*np.exp(-0.5 * distances_sq/l)
    return K

@njit(parallel=True)
def _get_global_kernel(features_N, features_M, symmetric, A, ls):
    N = len(features_N)
    M = len(features_M)
    K = np.zeros((N, M))
    l = ls*ls
    if not symmetric:
        for i in range(N):
            for j in range(M):
                K[i][j] = _global_covariance_fast(features_N[i], features_M[j], A=A, ls=ls)

        
    elif symmetric:
        for i in prange(N):
            for j in range(i, N):
                K[i][j] = _global_covariance_fast(features_N[i], features_M[j], A=A, ls=ls)
                if i != j:
                    K[j][i] = K[i][j]
    return K

@njit
def _get_global_kernel_derivative(features, A, ls):
    N = len(features)
    K = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i, N):
            K[i][j] = _global_covariance_derivative_fast(features[i], features[j], A=A, ls=ls)
            if i != j:
                K[j][i] = K[i][j]
    return K

@njit
def _global_covariance(X, Y, A, ls):
    l = ls*ls
    N_X = len(X)
    N_Y = len(Y)
    k = 0
    for ix in range(N_X):
        for iy in range(N_Y):
            d = 0
            for i in range(len(X[ix])):
                d += (X[ix][i] - Y[iy][i])*(X[ix][i] - Y[iy][i])/l
            k += A*A*np.exp(-0.5 * d)
    return k

@njit
def _global_covariance_fast(X, Y, A, ls):
    l = ls*ls
    A2 = A*A
    N_X = len(X)
    N_Y = len(Y)

    tmp_X2 = np.empty(N_X)
    for ix in range(N_X):
        s = 0
        for j in range(len(X[ix])):
            s += X[ix][j]*X[ix][j]
        tmp_X2[ix] = s

    tmp_Y2 = np.empty(N_Y)
    for iy in range(N_Y):
        s = 0
        for j in range(len(Y[iy])):
            s += Y[iy][j]*Y[iy][j]
        tmp_Y2[iy] = s

    prod = np.dot(X, Y.T)

    k = 0
    for ix in range(N_X):
        for iy in range(N_Y):
            #dist = -2.*prod[ix][iy]+tmp_X2[ix]+tmp_Y2[iy]
            dist = np.sum((X[ix]-Y[iy])**2)
            k += A2*np.exp(-0.5 * dist/l)
    return k


@njit
def _global_covariance_derivative(X, Y, A, ls):
    l = ls*ls
    N_X = len(X)
    N_Y = len(Y)
    k = 0
    for ix in range(N_X):
        for iy in range(N_Y):
            d = 0
            for i in range(len(X[ix])):
                d += (X[ix][i] - Y[iy][i])*(X[ix][i] - Y[iy][i])/l
            k += A*A/ls*np.exp(-0.5 * d)*d
    return k


@njit
def _global_covariance_derivative_fast(X, Y, A, ls):
    l = ls*ls
    A2 = A*A/ls**3
    N_X = len(X)
    N_Y = len(Y)

    tmp_X2 = np.empty(N_X)
    for ix in range(N_X):
        s = 0
        for j in range(len(X[ix])):
            s += X[ix][j]*X[ix][j]
        tmp_X2[ix] = s

    tmp_Y2 = np.empty(N_Y)
    for iy in range(N_Y):
        s = 0
        for j in range(len(Y[iy])):
            s += Y[iy][j]*Y[iy][j]
        tmp_Y2[iy] = s

    prod = np.dot(X, Y.T)

    k = 0
    for ix in range(N_X):
        for iy in range(N_Y):
            dist = -2.*prod[ix][iy]+tmp_X2[ix]+tmp_Y2[iy]
            k += A2*np.exp(-0.5 * dist/l)*dist
    return k



@njit
def _local_kernel(X, Y, A, ls):
    l = ls*ls
    N_X = len(X)
    N_Y = len(Y)    
    k = np.zeros((N_X, N_Y))
    for ix in range(N_X):
        for iy in range(N_Y):
            d = 0
            for i in range(len(X[ix])):
                d += (X[ix][i] - Y[iy][i])*(X[ix][i] - Y[iy][i])/l
            k[ix][iy] = A*A*np.exp(-0.5 * d)
    return k


class RBFKernel(KernelBaseClass):
    name = 'RBFKernel'
    
    def __init__(self, A=1, ls=10, ls_bounds=(1e-2, 1000)):
        self.A = A
        
        self.hyperparameter_bounds = [ls_bounds]
        self.hyperparameters = np.array([ls])

        
    def _set_hyperparameters(self, hyperparameters):
        self._ls = hyperparameters[0]

    @property
    def ls(self):
        return self._ls
    @ls.setter
    def ls(self, val):
        self._hyperparameters = np.array([val])
        self._ls = val
        

    def _get_global_kernel(self, features_N, features_M, symmetric=False):
        """
        Don't call this directly; us  __call__ method.

        The global kernel is the covariance between the atoms-objects
        kernel size: |data_N| x |data_M| 
        |x| = len(x)
        features_N: num data x num atoms in data x len feature
        """
        if not isinstance(features_N,list) and features_N.shape[1] == 1 and features_M.shape[1] == 1:
            X = features_N.reshape((features_N.shape[0],features_N.shape[2]))
            Y = features_M.reshape((features_M.shape[0],features_M.shape[2]))
            return _get_global_kernel_1x1_opt(X, Y, symmetric, self.A, self.ls)
        if not isinstance(features_N,list):
            if features_N.shape[1] == 1:
                print('KER1',features_N.shape,features_M.shape)
                X = features_N.reshape((features_N.shape[0],features_N.shape[2]))
                Y = features_M.reshape((features_M.shape[0]*features_M.shape[1],features_M.shape[2]))
                K = _get_global_kernel_1x1_opt(X, Y, symmetric, self.A, self.ls)
                K.shape = (K.shape[0],features_M.shape[0],features_M.shape[1])
                K = np.sum(K,axis=2)
                return K
            print('KER2',features_N.shape,features_M.shape)
        else:
            print('KER3',len(features_N),len(features_M))
        return _get_global_kernel(features_N, features_M, symmetric, self.A, self.ls)

    def get_kernel_gradient(self, features):
        if len(features.shape) == 3: # non-jagged!
            jac = _get_global_kernel_derivative(features, self.A, self.ls)
        else: # jagged - make awkward
            jagged_features = [a for a in features] #ak.Array([a for a in features])
            jac = _get_global_kernel_derivative(jagged_features, self.A, self.ls)
        jac = np.expand_dims(jac, axis=0)
        return jac

    def get_local_kernel(self, features_N, features_M):
        return _local_kernel(features_N, features_M, A=self.A, ls=self.ls)
