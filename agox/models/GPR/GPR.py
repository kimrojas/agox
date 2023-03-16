import numpy as np

from scipy.linalg import cho_solve, cho_factor
from scipy.optimize import fmin_l_bfgs_b

from ase.calculators.calculator import all_changes

from agox.models.ABC_model import ModelBaseClass
from agox.utils import candidate_list_comprehension

from agox.models.GPR.sparsifiers.CUR import CUR
from agox.utils.ray_utils import RayPoolUser, ray_kwarg_keys


class GPR(ModelBaseClass, RayPoolUser):

    name = 'GPR'

    implemented_properties = ['energy', 'forces', 'uncertainty', 'force_uncertainty']

    dynamic_attributes = ['alpha', 'K_inv', 'X', 'K', 'kernel', 'Y']

    """


    Attributes
    ----------
    descriptor : DescriptorBaseClass
        Descriptor object.
    kernel : KernelBaseClass
        Kernel object.
    descriptor_type : string
        Descriptor type.
    prior : ModelBaseClass
        Prior model object.
    sparsifier : SparsifierBaseClass
        Sparsifier object.
    single_atom_energies : dict
        Dictionary of single atom energies.
    use_prior_in_training : bool
        Whether to use prior in training.
    sparsifier : SparsifierBaseClass
        Sparsifier object.
    centralize : bool
        Whether to centralize the energy.
    alpha : np.ndarray
        Model parameters.
    X : np.ndarray
        Training features.
    Y : np.ndarray
        Training targets.
    update : bool
        Whether to update the model.
    prior_energy : np.ndarray
        Prior energy.
    mean_energy : float
        Mean energy.
    
    Methods
    -------
    train_model(training_data, **kwargs)
        Train the model.
    predict_energy(atoms, **kwargs)
        Predict the energy of a given structure.
    predict_uncertainty(atoms, **kwargs)
        Predict the uncertainty of a given structure.
    predict_forces(atoms, return_uncertainty=False, **kwargs)
        Predict the forces of a given structure.
    predict_forces_uncertainty(atoms, **kwargs)
        Predict the uncertainty of the forces of a given structure.


    """

    def __init__(self, descriptor, kernel, descriptor_type='global', prior=None, sparsifier=None,
                 single_atom_energies=None, use_prior_in_training=False,
                 sparsifier_cls=CUR, sparsifier_args=(1000,), sparsifier_kwargs={},
                 n_optimize=1, optimizer_maxiter=100, centralize=True, **kwargs):

        """
        Parameters
        ----------
        descriptor : DescriptorBaseClass
            Descriptor object.
        kernel : KernelBaseClass
            Kernel object.
        feature_method : function
            Feature method.
        prior : ModelBaseClass
            Prior model object.
        sparsifier_cls : SparsifierBaseClass
            Sparsifier object
        sparsifier_args : tuple
            Arguments for the sparsifier
        sparsifier_kwargs : dict
            Keyword arguments for the sparsifier
        single_atom_energies : dict
            Dictionary of single atom energies.
        use_prior_in_training : bool
            Whether to use prior in training.
        sparsifier : SparsifierBaseClass
            Sparsifier object

        centralize : bool
            Whether to centralize the energy.

        """        
        ray_kwargs = {key:kwargs.pop(key, None) for key in ray_kwarg_keys}
        ModelBaseClass.__init__(self, **kwargs)
        RayPoolUser.__init__(self, **ray_kwargs)

        self.descriptor = descriptor
        self.kernel = kernel
        self.feature_method = getattr(self.descriptor, 'get_' + descriptor_type + '_features')
        self.feature_gradient_method = getattr(self.descriptor, 'get_' + descriptor_type + '_feature_gradient')
        self.prior = prior
        self.sparsifier = sparsifier
        self.single_atom_energies = single_atom_energies
        self.use_prior_in_training = use_prior_in_training
        self.sparsifier = sparsifier_cls(*sparsifier_args, **sparsifier_kwargs)
        self.n_optimize = n_optimize
        self.optimizer_maxiter = optimizer_maxiter
        self.centralize = centralize

        self.add_save_attributes(['X', 'Y', 'mean_energy', 'K', 'K_inv', 'alpha', 'kernel.theta'])
        # Initialize all possible model parameters
        self.X = None
        self.Y = None
        self.K = None
        self.K_inv = None        
        self.alpha = None
        self.mean_energy = 0.

        # We add self to the pool, it will keep an updated copy of the model on the pool
        if self.use_ray:
            self.actor_model_key = self.pool_add_module(self)
            self.self_synchronizing = True # Defaults to False, inherited from Module.


    def model_info(self, **kwargs):
        """
        List of strings with model information
        """
        x = '    '
        sparsifier_name = self.sparsifier.name if self.sparsifier is not None else 'None'
        sparsifier_mpoints = self.sparsifier.m_points if self.sparsifier is not None else 'None'
        out =  ['------ Model Info ------',
                'Descriptor:',
                x + '{}'.format(self.descriptor.name),
                'Kernel:',
                x + '{}'.format(self.kernel),
                'Sparsifier:',
                x + '{} selecting {} points'.format(sparsifier_name, sparsifier_mpoints),
                '------ Training Info ------',
                'Training data size: {}'.format(self.X.shape[0]),]

        return out

            
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
        f = self.feature_method(atoms)

        if isinstance(f, np.ndarray) and len(f.shape) == 1:
            f = f.reshape(1, -1)
        f = np.vstack(f)
        
        return f

    def _train_model(self):
        """
        Train the model
        
        """
        if self.sparsifier is not None:
            self.X, m_idx = self.sparsifier(self.X)
            self.Y = self.Y[m_idx]
        
        if self.use_ray:
            self.pool_synchronize(attributes=['X', 'Y'], writer=self.writer)
            self.hyperparameter_search_parallel()
        else:
            self.hyperparameter_search()

        self.K = self.kernel(self.X)
        self.alpha, self.K_inv, _ = self._solve(self.K, self.Y)

        if self.use_ray:
            self.pool_synchronize(attributes=['alpha', 'K_inv', 'kernel', 'K'], writer=self.writer)
        
    def train_model(self, training_data, **kwargs):
        """
        Train the model.

        Parameters
        ----------
        training_data : list
            List of Atoms objects.

        """
        if self.update:
            self.X, self.Y = self._update(training_data, **kwargs)
        else:
            self.X, self.Y = self._preprocess(training_data)

        self._training_record(training_data)
  
        self._train_model()

        validation = self.validate()

        self.print_model_info(validation=validation)
        self.atoms = None
        self.ready_state = True

    @candidate_list_comprehension
    def predict_energy(self, atoms, k=None, **kwargs):
        if self.alpha is None:
            return self.postprocess_energy(atoms, 0)
        
        x = self.get_features(atoms)
        if k is None:
            k = self.kernel(self.X, x)

        e_pred = np.sum(k.T @ self.alpha)
        return self.postprocess_energy(atoms, e_pred)

    @candidate_list_comprehension        
    def predict_uncertainty(self, atoms, k=None, k0=None, **kwargs):
        """
        Predict uncertainty for a given atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        float
            Uncertainty
        
        """
        if 'uncertainty' not in self.implemented_properties or self.alpha is None:
            return 0
        
        x = self.get_features(atoms)
        if k is None:
            k = self.kernel(self.X, x)
        if k0 is None:
            k0 = self.kernel(x, x)
        var = float(k0 - k.T @ self.K_inv @ k)
        return np.sqrt(max(var, 0))

    @candidate_list_comprehension    
    def predict_forces(self, atoms, dkdr=None, **kwargs):
        """
        Predict forces for a given atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        np.ndarray
            Forces
        """
        if self.alpha is None:
            return self.postprocess_forces(atoms, np.zeros((len(atoms), 3)))
        
        # F_i = - dE / dr_i = dE/dk dk/df df/dr_i = - alpha dk/df df_dr_i
        x = self.get_features(atoms)
        if dkdr is None:
            dfdr = np.array(self.feature_gradient_method(atoms))
            dkdf = self.kernel.get_feature_gradient(self.X, x)
            dkdr = np.dot(dkdf, dfdr.T)
            
        f_pred = -np.dot(dkdr.T, self.alpha).reshape(-1,3)

        return self.postprocess_forces(atoms, f_pred)

    @candidate_list_comprehension        
    def predict_forces_uncertainty(self, atoms, dkdr=None, **kwargs):
        """
        Predict forces uncertainty for a given atoms object

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        np.ndarray
            Forces uncertainty
        
        """
        if 'forces_uncertainty' not in self.implemented_proporties or self.alpha is None:
            return np.zeros((len(atoms), 3))
        
        x = self.get_features(atoms)
        
        if dkdr is None:
            dfdr = np.array(self.feature_gradient_method(atoms))
            dkdf = self.kernel.get_feature_gradient(self.X, x)
            dkdr = np.dot(dkdf, dfdr.T)

        if k is None:
            k = self.kernel(self.X, x)
        if k0 is None:
            k0 = self.kernel(x, x)
            
        var = k0 - k.T @ self.K_inv @ k
        if var < 0:
            return np.zeros((len(atoms), 3))
        else:
            return 1/np.sqrt(var) * dkdr.T @ self.K_inv @ k


    def converter(self, atoms, reduced=False, **kwargs):
        """
        Precompute all necessary quantities for the model

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object

        Returns
        -------
        dict
            Dictionary with all necessary quantities
        
        """
        x = self.get_features(atoms)
        k = self.kernel(self.X, x)
        k0 = self.kernel(x, x)
        if reduced:
            return {'x': x, 'k': k, 'k0': k0,}
        
        dfdr = np.array(self.feature_gradient_method(atoms))
        dkdf = self.kernel.get_feature_gradient(self.X, x)
        dkdr = np.dot(dkdf, dfdr.T)

        return {'x': x, 'k': k, 'k0': k0, 'dkdr': dkdr}


    
    @property
    def single_atom_energies(self):
        """
        Get the single atom energies.

        Returns
        -------
        np.ndarray

        """
        return self._single_atom_energies
    
    @single_atom_energies.setter
    def single_atom_energies(self, s):
        """
        Set the single atom energies.
        Index number corresponds to the atomic number ie. 1 = H, 2 = He, etc.

        Parameters
        ----------
        s : dict or np.ndarray
            Dictionary/array of single atom energies.

        """
        if isinstance(s, np.ndarray):
            self._single_atom_energies = s
        elif isinstance(s, dict):
            self._single_atom_energies = np.zeros(100)
            for i, val in s.items():
                self._single_atom_energies[i] = val
        elif s is None:
            self._single_atom_energies = np.zeros(100)

    def _preprocess(self, data):
        """
        Preprocess the training data.

        Parameters
        ----------
        data : list
            List of Atoms objects.

        Returns
        -------
        np.ndarray
            Features.
        np.ndarray
            Targets.
        
        """
        Y = np.expand_dims(np.array([d.get_potential_energy() for d in data]), axis=1)
        
        if self.prior is None or not self.use_prior_in_training:
            self.prior_energy = np.zeros(Y.shape)
        else:
            self.prior_energy = np.expand_dims(np.array([self.prior.predict_energy(d) for d in data]), axis=1)

        Y -= self.prior_energy
        
        if self.centralize:
            self.mean_energy = np.mean(Y)
            
        Y -= self.mean_energy
        X = self.get_features(data)

        return X, Y

    def _update(self, training_data):
        """
        Update the the features and targets.
        training_data is all the data.

        Parameters
        ----------
        training_data : list
            List of Atoms objects.

        Returns
        -------
        np.ndarray
            Features.
        np.ndarray
            Targets.
        
        """
        return self._preprocess(training_data)
        
    def postprocess_energy(self, atoms, e_pred):
        """
        Postprocess the energy.

        Parameters
        ----------
        atoms : Atoms
            Atoms object.
        e_pred : float
            Predicted energy.

        Returns
        -------
        float
            Postprocessed energy.
        """
        if self.prior is None:
            prior = 0
        else:
            prior = self.prior.predict_energy(atoms)
            
        return e_pred + prior + self.mean_energy

    def postprocess_forces(self, atoms, f_pred):
        """
        Postprocess the forces.

        Parameters
        ----------
        atoms : Atoms
            Atoms object.
        f_pred : np.ndarray
            Predicted forces.

        Returns
        -------
        np.ndarray
            Postprocessed forces.
        """
        if self.prior is None:
            prior = 0
        else:
            prior = self.prior.predict_forces(atoms)
            
        return f_pred + prior
            
    def hyperparameter_search(self):
        """
        Hyperparameter search
        
        """
        initial_parameters = []
        initial_parameters.append(self.kernel.theta.copy())
        if self.n_optimize > 0:
            for _ in range(self.n_optimize-1):
                init_theta = np.random.uniform(size=(len(self.kernel.bounds),), low=self.kernel.bounds[:,0], high=self.kernel.bounds[:,1])
                initial_parameters.append(init_theta)

            fmins = []
            thetas = []
            for init_theta in initial_parameters:
                theta_min, nll_min = self._hyperparameter_optimize(init_theta=init_theta)
                fmins.append(nll_min)
                thetas.append(theta_min)

            self.kernel.theta = thetas[np.argmin(np.array(fmins))]

    def hyperparameter_search_parallel(self):
        """
        Hyperparameter search in parallel

        Parameters
        ----------
        update_actors : bool, optional
            Update the actors with the new kernel, by default True
        """

        N_jobs = self.cpu_count
        modules = [[self.actor_model_key]] * N_jobs # All jobs use the same model that is already on the actor. 
        args = [[self.n_optimize] for _ in range(N_jobs)] # Each job gets a different initial theta
        kwargs = [{} for _ in range(N_jobs)] # No kwargs
        kwargs[0]['use_current_theta'] = True # Use the current theta for the first job for one iteration.

        # Run the jobs in parallel
        outputs = self.pool_map(ray_hyperparameter_optimize, modules, args, kwargs)

        # Get the best theta
        likelihood = [output[1] for output in outputs]
        best_theta = outputs[np.argmin(likelihood)][0]
        
        # Set the best theta
        self.kernel.theta = best_theta

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        if 'uncertainty' in properties:
            self.results['uncertainty'] = self.predict_uncertainty(atoms)
        if 'force_uncertainty' in properties:
            self.results['force_uncertainty'] = self.predict_forces_uncertainty(atoms)


            
    def _hyperparameter_optimize(self, init_theta=None):
        """
        Hyperparameter optimization

        Parameters
        ----------
        init_theta : np.ndarray
            Initial theta

        Returns
        -------
        np.ndarray
            Optimal theta
        
        """
        def f(theta):
            P, grad_P = self._marginal_log_likelihood_gradient(theta)
            if np.isnan(P):
                return np.inf, np.zeros_like(theta, dtype='float64')
            P, grad_P = -float(P), -np.asarray(grad_P, dtype='float64')
            return P, grad_P

        bounds = self.kernel.bounds
        
        if init_theta is None:
            self.key, key = random.split(self.key)
            init_theta = random.uniform(key, shape=(len(bounds),), minval=bounds[:,0], maxval=bounds[:,1])
            
        theta_min, fmin, conv = fmin_l_bfgs_b(f, np.asarray(init_theta, dtype='float64'),
                                              bounds=np.asarray(bounds, dtype='float64'),
                                              maxiter=self.optimizer_maxiter)

        return theta_min, fmin
    
    def _solve(self, K, Y):
        """
        Solve the linear system

        Parameters
        ----------
        K : np.ndarray
            Kernel matrix
        Y : np.ndarray
            Target values

        Returns
        -------
        np.ndarray
            Alpha
        np.ndarray
            Inverse of kernel matrix
        np.ndarray
            Cholesky decomposition of kernel matrix
        
        """
        L, lower = cho_factor(K)
        alpha = cho_solve((L, lower), Y)
        K_inv = cho_solve((L, lower), np.eye(K.shape[0]))
        return alpha, K_inv, (L, lower)
    
    def _marginal_log_likelihood(self, theta):
        """
        Marginal log likelihood

        Parameters
        ----------
        theta : np.ndarray
            Kernel parameters

        Returns
        -------
        float
            Marginal log likelihood
        
        """
        t = self.kernel.theta.copy()
        self.kernel.theta = theta        
        K = self.kernel(self.X)
        self.kernel.theta = t        
        
        alpha, K_inv, (L, lower) = self._solve(K, self.Y)

        log_P = - 0.5 * np.einsum("ik,ik->k", self.Y, alpha) \
            - np.sum(np.log(np.diag(L))) \
            - K.shape[0] / 2 * np.log(2 * np.pi)

        return np.sum(log_P)

    def _marginal_log_likelihood_gradient(self, theta):
        """
        Marginal log likelihood gradient

        Parameters
        ----------
        theta : np.ndarray
            Kernel parameters

        Returns
        -------
        float
            Marginal log likelihood
        np.ndarray
            Marginal log likelihood gradient

        """
        t = self.kernel.theta.copy()
        self.kernel.theta = theta
        K, K_hp_gradient = self.kernel(self.X, eval_gradient=True)
        self.kernel.theta = t
        
        alpha, K_inv, (L, lower) = self._solve(K, self.Y)

        log_P = - 0.5 * np.einsum("ik,ik->k", self.Y, alpha) \
            - np.sum(np.log(np.diag(L))) \
            - K.shape[0] / 2 * np.log(2 * np.pi)
        
        inner = np.squeeze(np.einsum("ik,jk->ijk", alpha, alpha), axis=2) - K_inv
        inner = inner[:,:,np.newaxis]
        
        grad_log_P = np.sum(0.5 * np.einsum("ijl,ijk->kl", inner, K_hp_gradient), axis=-1)
        return log_P, grad_log_P
    

    
def ray_hyperparameter_optimize(model, n_opt, use_current_theta=False):
    """
    Hyperparameter optimization

    Parameters
    ----------
    init_theta : np.ndarray
        Initial theta

    Returns
    -------
    np.ndarray
        Optimal theta
    
    """
    def f(theta):
        P, grad_P = model._marginal_log_likelihood_gradient(theta)
        if np.isnan(P):
            return np.inf, np.zeros_like(theta, dtype='float64')
        P, grad_P = -float(P), -np.asarray(grad_P, dtype='float64')
        return P, grad_P

    def init_theta_func(bounds):
        return np.random.uniform(size=len(bounds,), low=bounds[:,0], high=bounds[:,1])
    
    bounds = model.kernel.bounds

    fbest = np.inf
    for i in range(n_opt):
        
        if not use_current_theta:
            init_theta = init_theta_func(bounds)
        else:
            init_theta = model.kernel.theta
            use_current_theta = False

        theta_min, fmin, conv = fmin_l_bfgs_b(f, np.asarray(init_theta, dtype='float64'),
                                                bounds=np.asarray(bounds, dtype='float64'),
                                                maxiter=model.optimizer_maxiter)
        if fmin < fbest:
            fbest = fmin
            theta_min_best = theta_min

    return theta_min, fmin



