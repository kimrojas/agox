import numpy as np
from agox.models.beacon.kerneltypes import EuclideanDistance, SquaredExp

class ICEDistance(EuclideanDistance):

    @staticmethod
    def dD_dfraction(fp1, fp2):
        ''' Gradient of distance function:

                      d D(x, x')
                      ----------
                         d q
        '''

        D = EuclideanDistance.distance(fp1, fp2)

        if D == 0.0:
            return np.zeros((len(fp1.frac_gradients), 1))

        # difference vector between fingerprints:
        diff = fp1.vector - fp2.vector

        result = 1 / D * np.einsum('i,hil->hl',
                                   diff,
                                   fp1.reduce_frac_gradients(),
                                   optimize=True)

        return result

class ICESquaredExp(SquaredExp):

    def __init__(self, weight=1.0, scale=1.0):

        SquaredExp.__init__(self, weight=weight, scale=scale)
        self.metric = ICEDistance

    def kernel_gradient_frac(self, fp1, fp2):
        """
        Kernel gradients w.r.t. fractions only
        """
        return self.dk_dD(fp1, fp2) * self.metric.dD_dfraction(fp1, fp2)

    def dkernelgradient_dq(self, fp1, fp2):
        '''
        Fraction derivatives of the kernel gradients (for coordinates).
        The derivative for q is for an atom in self, whereas
        the gradients are from fp2.
        '''

        # if fp1.emptyfingerprint:
        #     return np.zeros((fp1.natoms, fp1.natoms, 3, 1))

        k = self.kernel(fp1, fp2)

        coord_grads = fp2.reduce_coord_gradients()

        C0 = np.einsum('hm,i,gin->hgnm',
                       self.kernel_gradient_frac(fp1, fp2),
                       (fp1.vector - fp2.vector),
                       coord_grads,
                       optimize=True)

        C1 = k * np.einsum('him,gin->hgnm',
                           fp1.reduce_frac_gradients(),
                           coord_grads,
                           optimize=True)

        result = 1 / self.scale**2 * (C0 + C1)

        return result

