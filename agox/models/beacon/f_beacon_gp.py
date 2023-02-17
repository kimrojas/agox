#from agox.models.beacon.beacon_gp import GaussianProcess
#from gpatom.beacon.icebeacon.gp.kernel import ICEKernel, ICEKernelNoforces
import numpy as np

from agox.models.beacon.beacon_gp import GaussianProcess

from agox.models.beacon.f_kernel import ICEKernel, ICEKernelNoforces

from ase.calculators.calculator import all_changes, Calculator


class ICEGaussianProcess(GaussianProcess):
    '''
    Gaussian process that is specifically used in ICEBEACON.
    The essential difference is that we need to predict
    the derivatives of the surrogate PES w.r.t. the fractions.
    '''

    def __init__(self, **kwargs):
        GaussianProcess.__init__(self, **kwargs)

        kernelparams = {'weight': self.hp['weight'],
                        'scale': self.hp['scale']}



        if self.use_forces:
            self.kernel = ICEKernel(kerneltype='sqexp', params=kernelparams)
        else:
            self.kernel = ICEKernelNoforces(kerneltype='sqexp', params=kernelparams)

    def calculate_prior_array(self, list_of_fingerprints, forces=True):
        if forces:
            return GaussianProcess.calculate_prior_array(self, list_of_fingerprints)

        self.prior.use_forces = False
        result = GaussianProcess.calculate_prior_array(self, list_of_fingerprints)
        self.prior.use_forces = self.use_forces
        return result

#    @property
#    def natoms(self, n_ghost):
#        return len(self.X[0].atoms)
#        #return len(self.X[0].atoms)+2   # plus the two missing atoms

#    @property
#    def ntrain(self):
#        return len(self.X)

#    @property
#    def nforces(self):          #
#        return 3 * self.natoms
#        #return 3 * (self.natoms-2)   # minus the two missing atoms

#    @property
#    def singlesize(self):
#        return 1 + self.nforces

#    @property
#    def alphasize(self):
#        return self.ntrain * self.singlesize


    def predict(self, atoms, fractions=None, get_variance=False, calc_frac_gradients=True, n_ghost=0):
        ''' If get_variance=False, then variance
        is returned as None '''
        
        x=self.descriptor.get_fp_object(atoms, fractions)
        
        #print('In predict')
        #print(x)
        #print(self.X)
        
        ######################
        self.natoms=len(self.X[0].atoms)+n_ghost
        self.ntrain=len(self.X)
        self.nforces=3 * (self.natoms-n_ghost)
        self.singlesize=1 + self.nforces
        self.alphasize=self.ntrain * self.singlesize
        #######################
        
        if self.use_forces:

            kv = self.kernel.kernel_vector(x, self.X)  # x is the full fingerprint.    self.X is the ghost fingerprint
            #assert kv.shape == (self.singlesize, self.alphasize)   

            prior_array = self.calculate_prior_array([x])
            f = prior_array + np.dot(kv, self.model_vector) # f is an energy and natomsx3 forces. 
                        
            
            coord_gradients = f[1:].reshape(self.natoms, 3)    #   just the natomsx3 forces.
        
        
            if calc_frac_gradients:
                frac_gradients = self.get_frac_gradients(x)
            else:
                frac_gradients = np.zeros([self.natoms,1])
            assert frac_gradients.shape == (self.natoms, 1)  
            
            
            all_gradients = np.concatenate((coord_gradients, frac_gradients),
                                           axis=1).flatten()
            
            
            f = [f[0]] + list(all_gradients)

            f = np.array(f)

            V = self.calculate_variance(get_variance, kv, x)


        else:
            k = self.kernel.kernel_vector(x, self.X)
            prior_array = self.calculate_prior_array([x])
            f = prior_array + np.dot(k, self.model_vector)

            dk_dxi = np.array([self.kernel.kerneltype.kernel_gradient(x, x2)
                               for x2 in self.X])
            dk_dq = np.array([self.kernel.kerneltype.kernel_gradient_frac(x, x2)
                               for x2 in self.X])

            dk = np.concatenate((dk_dxi, dk_dq), axis=2)

            forces = np.einsum('ijk,i->jk', dk,
                               self.model_vector).flatten()

            f = list(f) + list(forces)
            f = np.array(f)

            V = self.calculate_variance(get_variance, k, x)
            

        return f, V

    def get_frac_gradients(self, x):

        dk_dq = np.array([self.kernel.kerneltype.kernel_gradient_frac(x, x2)
                          for x2 in self.X])
    
        
        #assert dk_dq.shape == (self.ntrain, self.natoms, 1)

        # Fractions derivatives of kernel gradients:
        d2k_drm_dq = np.array([self.kernel.kerneltype.dkernelgradient_dq(x, x2)
                               for x2 in self.X])

        #assert d2k_drm_dq.shape == (self.ntrain, self.natoms, self.natoms, 3, 1)


        d2k_drm_dq = d2k_drm_dq.reshape(self.ntrain, self.natoms, self.nforces)   

        # Kernel vector:
        K_x_X = np.concatenate((dk_dq, d2k_drm_dq), axis=2)     
        K_x_X = K_x_X.swapaxes(0, 1)
        K_x_X = K_x_X.reshape((self.natoms, self.alphasize, 1))   

        
        frac_gradients = (np.einsum('ijk,j->ik', K_x_X, self.model_vector,
                                    optimize=True))
        
        return frac_gradients



    def get_properties(self, atoms):
        
        
        if hasattr(atoms, 'fractions'):
            fractions=atoms.fractions
            n_ghost=atoms.n_ghost
        else:
            fractions=None
            n_ghost=0
            
        
        
        f, V = self.predict(atoms, fractions, get_variance=True, n_ghost=n_ghost)
        energy=f[0]
        grads=f[1:].reshape(-1, 4)
        forces=-1*grads[:, :3]
        
       # print(np.shape(V))
        
        frac_grads=grads[:, 3]
        
        if self.use_forces:
            
            print(V[0,0])
            
            uncertainty_energy = np.sqrt(V[0, 0])
            uncertainty_forces= np.sqrt(   np.diag(V[1:,1:])  ).reshape(-1, 3)
        else:
            uncertainty_energy=np.sqrt(V[0])
            uncertainty_forces=np.zeros_like(forces)
        
        return energy, forces, uncertainty_energy, uncertainty_forces, frac_grads
    
    

    def predict_energy(self, atoms, return_uncertainty=False, **kwargs):
        
        energy, forces, unc_energy, unc_forces, frac_grads = self.get_properties(atoms)

        if return_uncertainty:
            return energy, unc_energy
        return energy


    def predict_forces(self, atoms, return_uncertainty=False, **kwargs):
        energy, forces, und_energy, unc_forces, frac_grads = self.get_properties(atoms)
        
        forces=np.hstack( (forces, -1*frac_grads.reshape(16,1) ) )
        
        unc_forces=np.hstack(  ( unc_forces, np.zeros_like(frac_grads.reshape(16,1)) )   )
    
        if return_uncertainty:
            return forces, unc_forces
        return forces    
    
    
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
       """ASE Calculator calculate method
       
       Parameters
       ----------
       atoms : ASE Atoms obj or AGOX Candidate object
           The atoms object for to predict properties of.
       properties : :obj: `list` of :obj: `str`
           List of properties to calculate for the atoms
       system_changes : ASE system_changes
           Which ASE system_changes to check for before calculation
       
       Returns
       ----------
       None
       """                   
       
       
       #print(atoms)
       
       Calculator.calculate(self, atoms, properties, system_changes)

#       self.atoms.fractions=atoms.fractions        
#       self.atoms.n_ghost=atoms.n_ghost 

        # doesnt work 

       E, sigma = self.predict_energy(atoms, return_uncertainty=True)
       #E, sigma = self.predict_energy(self.atoms, return_uncertainty=True)       # is it necessary that it is self.atoms???
       self.results['energy'] = E
       
       if 'forces' in properties:
      #     forces = self.predict_forces(self.atoms)
           forces = self.predict_forces(atoms)
           self.results['forces'] = forces
       
       self.results['uncertainty'] = sigma
      # self.results['force_uncertainties'] = 0
    
    
    
    
    
    
    