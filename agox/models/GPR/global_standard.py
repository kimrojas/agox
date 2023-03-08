import warnings
import numpy as np
from agox.models.GPR.ABC_standard import StandardBaseClass


class GlobalGPR(StandardBaseClass):

    name = 'GlobalGPR'

    def get_features(self, atoms):
        """
        Get features for a given atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        np.ndarray
            Features
        
        """
        f = np.array(self.descriptor.get_global_features(atoms))
        if len(f.shape) == 1:
            f = f.reshape(1, -1)
        return f

    def get_model_parameters(self):
        warnings.warn('get_model_parameters is deprecated and will be removed in a future release.', DeprecationWarning)        
        parameters = {}
        parameters['feature_mat'] = self.X
        parameters['alpha'] = self.alpha
        parameters['bias'] = self.mean_energy
        parameters['kernel_hyperparameters'] = self.kernel.get_params()
        parameters['K_inv'] = self.K_inv
        parameters['iteration'] = self.get_iteration_counter()
        return parameters
    
    def set_model_parameters(self, parameters):
        warnings.warn('set_model_parameters is deprecated and will be removed in a future release.', DeprecationWarning)
        self.X = parameters['feature_mat']
        self.alpha = parameters['alpha']
        self.mean_energy = parameters['bias']
        self.K_inv = parameters['K_inv']
        self.set_iteration_counter(parameters['iteration'])
        self.kernel.set_params(**parameters['kernel_hyperparameters'])
        self.ready_state = True


    def get_feature_calculator(self):
        warnings.warn("The 'get_feature_calculator'-method will be deprecated in a future release.", DeprecationWarning)
        return self.descriptor

        
    @classmethod
    def default(cls, environment=None, database=None, temp_atoms=None, lambda1min=1e-1, lambda1max=1e3, lambda2min=1e-1, lambda2max=1e3, 
                theta0min=1, theta0max=1e5, beta=0.01, use_delta_func=True, sigma_noise=1e-2,
                descriptor=None, kernel=None, max_iterations=9999, max_training_data=1000):

        """
        Creates a GPR model. 

        Parameters
        ------------
        environment: AGOX environment. 
            Used to create an atoms object to initialize e.g. the feature calculator. 
        database: 
            AGOX database that the model will be attached to. 
        lambda1min/lambda1max: float
            Length scale minimum and maximum. 
        lambda2min/lambda2max: float
            Length scale minimum and maximum. 
        theta0min/theta0max: float
            Amplitude minimum and maximum 
        use_delta_func: bool
            Whether to use the repulsive prior function. 
        sigma_noise: float
            Noise amplitude. 
        feature_calculator: object
            A feature calculator object, if None defaults to reasonable 
            fingerprint settings. 
        kernel: str or kernel object or None. 
            If kernel='anisotropic' the anisotropic RBF kernel is used where
            radial and angular componenets are treated at different length scales
            If None the standard RBF is used. 
            If a kernel object then that kernel is used. 
        max_iterations: int or None
            Maximum number of iterations for the hyperparameter optimization during 
            its BFGS optimization through scipy. 
        """
        
        from ase import Atoms
        from agox.models.priors.repulsive import Repulsive
        from agox.models.GPR.kernels import RBF, Constant as C, Noise

        assert temp_atoms is not None or environment is not None

        if temp_atoms is None:
            temp_atoms = environment.get_template()
            temp_atoms += Atoms(environment.get_numbers())

        if descriptor is None:
            from agox.models.descriptors import Fingerprint
            descriptor = Fingerprint(temp_atoms, use_cache=True)

        lambda1ini = (lambda1max - lambda1min)/2 + lambda1min
        lambda2ini = (lambda2max - lambda2min)/2 + lambda2min
        theta0ini = 5000                         
        
        if kernel is None:
            kernel = C(theta0ini, (theta0min, theta0max)) * \
            ( \
            C((1-beta), ((1-beta), (1-beta))) * RBF(lambda1ini, (lambda1min,lambda1max)) + \
            C(beta, (beta, beta)) * RBF(lambda2ini, (lambda2min,lambda2max)) 
            ) + \
            Noise(sigma_noise, (sigma_noise,sigma_noise))
            

        if use_delta_func:
            delta = Repulsive(rcut=6)
        else:
            delta = None
        
        return cls(database=database, kernel=kernel, descriptor=descriptor, prior=delta,
                   n_optimize=1, optimizer_maxiter=max_iterations)


