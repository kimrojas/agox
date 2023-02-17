import numpy as np
#from ase.parallel import world
from agox.models.beacon.f_kerneltypes import ICESquaredExp
from agox.models.beacon.kernel import FPKernelParallel


class ICEKernel(FPKernelParallel):

    def __init__(self, kerneltype='sqexp', params=None):
        '''
        hp: dict
            Hyperparameters for the kernel type
        '''
        kerneltypes = {'sqexp': ICESquaredExp}

        if params is None:
            params = {}

        kerneltype = kerneltypes.get(kerneltype)
        self.kerneltype = kerneltype(**params)
        self.ndof = 3



    def kernel_function_hessian(self, x1, x2):
        hessian = self.kerneltype.kernel_hessian(x1, x2)

        # Reshape to 2D matrix:
        size = self.get_size(x1) - 1
        size2 = self.get_size(x2) - 1      # CL:  made by me
        
        #hessian = hessian.swapaxes(1, 2).reshape(size, size)
        hessian = hessian.swapaxes(1, 2).reshape(size, size2) # CL made by me instead of the commented out line
        return hessian

    def kernel(self, x1, x2):
        '''
        This function returns a D+1 x D+1 matrix, where D is
        the dimension of the manifold
        
        CL:  modified to yield a D1+1 x D2+1 matrix. 
        '''

        size = self.get_size(x1)
        size2=self.get_size(x2)   # CL made by me
                
        #K = np.empty((size, size), dtype=float) 
        K = np.zeros((size, size2), dtype=float)    # changed from empty to zeros just to be easier to look at


        K[0, 0] = self.kerneltype.kernel(x1, x2)
        
        K[1:, 0] = self.kernel_function_gradient(x1, x2)
        
        K[0, 1:] = self.kernel_function_gradient(x2, x1)
        
        K[1:, 1:] = self.kernel_function_hessian(x1, x2)

        return K

 



class ICEKernelNoforces(ICEKernel):

    def kernel(self, x1, x2):
        return np.atleast_1d(self.kerneltype.kernel(x1, x2))

    def get_size(self, x):
        '''
        Return the size of a kernel matrix
        x: fingerprint
        '''
        return 1
