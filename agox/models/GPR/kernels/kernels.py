import numpy as np
from sklearn.gaussian_process.kernels import RBF as sklearn_RBF
from sklearn.gaussian_process.kernels import ConstantKernel as sklearn_ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel as sklearn_WhiteKernel
from sklearn.gaussian_process.kernels import Sum as sklearn_Sum
from sklearn.gaussian_process.kernels import Product as sklearn_Product



class Kernel():
    def __add__(self, b):
        if not isinstance(b, Kernel):
            return Sum(self, Constant(b))
        return Sum(self, b)

    def __radd__(self, b):
        if not isinstance(b, Kernel):
            return Sum(Constant(b), self)
        return Sum(b, self)

    def __mul__(self, b):
        if not isinstance(b, Kernel):
            return Product(self, Constant(b))
        return Product(self, b)

    def __rmul__(self, b):
        if not isinstance(b, Kernel):
            return Product(Constant(b), self)
        return Product(b, self)

    
class RBF(Kernel, sklearn_RBF):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-1, 1e3)):
        sklearn_RBF.__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)

    
    def get_feature_gradient(self, X, Y):
        if np.ndim(Y) == 1:
            x = x[np.newaxis, :]
            
        K = self(X, Y)
        dK_dd = -1 / (2 * self.length_scale ** 2) * K
        dd_df = -2 * (X - Y)
        dk_df = np.multiply(dK_dd, dd_df)
        
        return dk_df

        
class Constant(Kernel, sklearn_ConstantKernel):
    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-1, 1e5)):
        sklearn_ConstantKernel.__init__(constant_value=constant_value, constant_value_bounds=constant_value_bounds)

        
    def get_feature_gradient(self, X, Y):
        return 0

class Noise(Kernel, sklearn_WhiteKernel):
    def __init__(self, noise_level=0.01, noise_level_bounds=(0.01, 0.01)):
        sklearn_WhiteKernel.__init__(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
        
    def get_feature_gradient(self, X, Y):
        return 0


class Sum(Kernel, sklearn_Sum):
    def get_feature_gradient(self, X, Y):
        return self.k1.get_feature_gradient(X,Y) + self.k2.get_feature_gradient(X,Y)

class Product(Kernel, sklearn_Product):
    def get_feature_gradient(self, X, Y):
        K1 = self.k1(X,Y)
        K2 = self.k2(X,Y)
        grad_K1 = self.k1.get_feature_gradient(X,Y)
        grad_K2 = self.k2.get_feature_gradient(X,Y)
        
        return K1 * grad_K2 + grad_K1 * K2
