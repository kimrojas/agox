import numpy as np
from sklearn.gaussian_process.kernels import RBF as sklearn_RBF
from sklearn.gaussian_process.kernels import ConstantKernel as sklearn_ConstantKernel
from sklearn.gaussian_process.kernels import WhiteKernel as sklearn_WhiteKernel


class RBF(sklearn_RBF):
    def get_feature_gradient(self, X, Y):
        if np.ndim(Y) == 1:
            x = x[np.newaxis, :]
            
        K = self(X, Y)
        dK_dd = -1 / (2 * self.length_scale ** 2) * K
        dd_df = -2 * (X - Y)
        dk_df = np.multiply(dK_dd, dd_df)
        
        return dk_df

        
class Constant(sklearn_ConstantKernel):
    def get_feature_gradient(self, X, Y):
        return 0

class Noise(sklearn_WhiteKernel):
    def get_feature_gradient(self, X, Y):
        return 0
