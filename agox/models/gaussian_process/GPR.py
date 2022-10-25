import numpy as np
#from copy import deepcopy
import pdb

import warnings
from operator import itemgetter

from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b
from .kernels import clone
import time

from agox.writer import agox_writer, Writer


class GPR(Writer):
    """
    comparator:
    Class to calculate similarities between structures based on their feature vectors.
    The comparator coresponds to the choice of kernel for the model

    featureCalculator:
    Class to calculate the features of structures based on the atomic positions of the structures.

    reg:
    Regularization parameter for the model

    comparator_kwargs:
    Parameters for the compator. This could be the width for the gaussian kernel.
    """
    def __init__(self, kernel, featureCalculator, delta_function=None, bias_func=None, optimize=True, n_restarts_optimizer=0, constraint_small_kernel_width=False,use_delta_in_training=False, 
                 verbose=True, n_maxiter_optimizer=None):
        Writer.__init__(self, verbose=verbose)

        if use_delta_in_training:
            print('#'*50)
            print('use_delta_in_training = True is NOT recommended, use at your own risk')
            print('#'*50)

        self.kernel = kernel
        self.featureCalculator = featureCalculator
        self.optimize = optimize
        self.n_restarts_optimizer = n_restarts_optimizer
        self.constraint_small_kernel_width = constraint_small_kernel_width
        self.use_delta_in_training=use_delta_in_training
        
        self.bias_func = bias_func
        self.delta_function = delta_function

        self.n_maxiter_optimizer = n_maxiter_optimizer

        # Initialize data counter
        self.Ndata = 0

    def predict_energy(self, atoms=None, fnew=None, K_vec=None, delta_value=None, return_error=False, no_prior_on_error=False):
        """
        Predict the energy of a new structure.
        """
        if K_vec is None:
            if fnew is None:
                fnew = self.featureCalculator.get_feature(atoms)
            K_vec = self.kernel_.get_kernel(self.featureMat, fnew).reshape(-1)

        if delta_value is None:
            if self.delta_function is not None:
                delta = self.delta_function.energy(atoms)
            else:
                delta = 0
        else:
            delta = delta_value

        Epred = K_vec.T.dot(self.alpha) + self.bias + delta

        if return_error:
            """
            v = cho_solve((self.L_, True), K_vec)  # Line 5
            K0 = self.kernel_.get_kernel(fnew,fnew)
            E_std = np.sqrt(K0 - K_vec.T.dot(v))  # Line 6
            #E_std = np.sqrt(self.kernel_(fnew) - K_vec.T.dot(v))  # Line 6
            return Epred, E_std, K0
            """
            
            alpha_err = np.dot(self.K_inv, K_vec)
            K0 = self.kernel_.get_kernel(fnew,fnew)
            g = K0 - np.dot(K_vec, alpha_err)
            g = max(0,g)  # Handle numerical errors. 
            E_std = np.sqrt(g)
            return Epred, E_std, K0
        else:
            return Epred
    
    def predict_force(self, atoms=None, fnew=None, fgrad=None, return_error=False):
        """
        Predict the force of a new structure.
        """
        
        # Calculate features and their gradients if not given
        if fnew is None:
            fnew = self.featureCalculator.get_feature(atoms)
        if fgrad is None:
            fgrad = self.featureCalculator.get_featureGradient(atoms)
        dk_df = self.kernel_.get_kernel_jac(self.featureMat, fnew)
        
        # Calculate contribution from delta-function
        if self.delta_function is not None:
            delta_force = self.delta_function.forces(atoms)
        else:
            Ncoord = 3 * len(atoms)
            delta_force = np.zeros(Ncoord)

        kernelDeriv = np.dot(dk_df, fgrad.T)
        F = -(kernelDeriv.T).dot(self.alpha) + delta_force
        
        if return_error:
            """
            K_vec = self.kernel_.get_kernel(self.featureMat, fnew)
            kernelDeriv = np.dot(dk_df, fgrad.T)
            v = cho_solve((self.L_, True), K_vec)  # Line 5
            error_force = np.sqrt(self.kernel_.get_kernel_jac(fnew,fnew) - kernelDeriv.T.dot(v))  # Line 6
            """

            K_vec = self.kernel_.get_kernel(self.featureMat, fnew).reshape(-1)
            alpha_err = np.dot(self.K_inv, K_vec)
            K0 = self.kernel_.get_kernel(fnew,fnew)
            g = K0 - np.dot(K_vec.T, alpha_err)  # negative g can occur due to numerical errors.
            if g <= 0:
                self.writer('negative g-value: g={}'.format(g))
                Ncoord = 3 * len(atoms)
                error_force = np.zeros(Ncoord)
            else:
                error_force = 1/np.sqrt(g) * (kernelDeriv.T).dot(alpha_err)
            return F, error_force
        else:
            return F

    def save_data(self, data_values_save, featureMat_save, delta_values_save=None, add_new_data=False):
        """
        Adds data to previously saved data.
        """
        Nsave = len(data_values_save)

        if Nsave > 0:
            if add_new_data and self.Ndata > 0:
                # Add data
                self.data_values = np.r_[self.data_values, data_values_save]
                self.featureMat = np.r_[self.featureMat, featureMat_save]
                if self.delta_function is not None and self.use_delta_in_training:
                    self.delta_values = np.r_[self.delta_values, delta_values_save]
                else:
                    self.delta_values = np.zeros(1)
                    
                # Iterate data counter
                self.Ndata += Nsave                    
            else:
                # Initialize data objects
                self.Ndata = len(data_values_save)
                self.data_values = data_values_save
                self.featureMat = featureMat_save
                if self.delta_function is not None and self.use_delta_in_training:
                    self.delta_values = delta_values_save
                else:
                    self.delta_values = np.zeros(1)

    def calc_bias(self, y):
        if self.bias_func is not None:
            if callable(self.bias_func):
                bias = self.bias_func(y)
            else:
                bias = self.bias_func
        else:
            bias = np.mean(y)

        return bias
        
    def train(self, atoms_list=None, data_values=None, features=None, delta_values=None, add_new_data=True, optimize=None, comm=None):
        """
        Train the model using gridsearch and cross-validation
            
        --- Input ---
        data_values:
        The labels of the new training data. In our case, the energies of the new training structures.

        featureMat:
        The features of the new training structures.

        positionMat:
        The atomic positions of the new training structures.

        add_new_data:
        If True, the data passed will be added to previously saved data (if any).

        k:
        Performs k-fold cross-validation.

        **GSkwargs:
        Dict containing the sequences of the kernel-width and regularization parameter to be
        used in grissearch. The labels are 'sigma' and 'reg' respectively.
        """
        
        if features is None:
            t = time.time()
            features = self.featureCalculator.get_featureMat(atoms_list)
            if self.verbose:
                self.writer('Feature time: %4.4f' %(time.time()-t))
            
        if data_values is None:
            data_values = np.array([atoms.get_potential_energy() for atoms in atoms_list])

        if delta_values is None:
            if self.delta_function is not None and self.use_delta_in_training:
                delta_values = np.array([self.delta_function.energy(a) for a in atoms_list])

        self.save_data(data_values_save=data_values,
                       featureMat_save=features,
                       delta_values_save=delta_values,
                       add_new_data=add_new_data)

        
        self.bias = self.calc_bias(self.data_values - self.delta_values)
        self.y_train = self.data_values - self.delta_values - self.bias

        if optimize is None:
            optimize = self.optimize
        
        if comm is not None:
            master = comm.rank == 0
        else:
            master = False

        try:
            self.kernel_ = clone(self.kernel_)
        except:
            self.kernel_ = clone(self.kernel)

        if optimize and self.kernel_.n_dims > 0:
            if self.verbose:
                self.writer('Optimizing log-likelihood - {}'.format(self.n_restarts_optimizer))
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            # First optimize starting from theta specified in kernel
            if comm is None or master:
                optima = [(self._constrained_optimization(obj_func,
                                                          self.kernel_.theta,
                                                          self.kernel_.bounds))]
            else:
                optima = []

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer - master):
                    theta_initial = \
                        np.random.uniform(bounds[:, 0], bounds[:, 1])
                    if self.constraint_small_kernel_width:
                        theta_initial[4] = np.random.uniform(bounds[4,0], theta_initial[2])
                    new_optimum = self._constrained_optimization(obj_func, theta_initial, bounds)
                    if self.constraint_small_kernel_width:
                        if new_optimum[0][4] < new_optimum[0][2]+1:
                            optima.append(new_optimum)
                    else:
                        optima.append(new_optimum)
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
            #self.writer('lml:', self.log_marginal_likelihood_value_, 'kernel_best:', self.kernel_)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)

        if comm is not None:
            hyperparam_results = [self.log_marginal_likelihood_value_, self.kernel_.theta]
            hyperparam_results_all = comm.gather(hyperparam_results, root=0)
            if master:
                lml_all = np.array([result[0] for result in hyperparam_results_all])
                index_best_theta = np.argmax(lml_all)
                results_best = hyperparam_results_all[index_best_theta]
            else:
                results_best = None

            results_best = comm.bcast(results_best, root=0)
            self.kernel_.theta = results_best[1]
            self.log_marginal_likelihood_value_ = results_best[0]

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.featureMat)
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            t = time.time()
            L_inv = np.linalg.inv(self.L_)
            if self.verbose:
                self.writer('Inversion time: %4.4f' %(time.time()-t))
            self.K_inv = L_inv.T @ L_inv
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha = cho_solve((self.L_, True), self.y_train)  # Line 3

        if np.any(np.isnan(self.alpha)):
            self.writer('alpha:\n', self.alpha, flush=True)
            
        if self.verbose:
            self.writer('GPR model k1:',self.kernel_.get_params().get('k1','NA'))
            self.writer('GPR model k2:',self.kernel_.get_params().get('k2','NA'))
        
        return 0,0
        #return self

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.
        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        kernel = self.kernel_.clone_with_theta(theta) # I think this dumb

        if eval_gradient:
            K, K_gradient = kernel(self.featureMat, eval_gradient=True)
        else:
            K = kernel(self.featureMat)

        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]
        
        alpha = cho_solve((L, True), y_train)  # Line 3

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)
        # self.writer(log_likelihood)
        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.n_maxiter_optimizer is None:
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
        else:
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, maxiter=self.n_maxiter_optimizer)
        if convergence_dict["warnflag"] != 0:
            warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                          " state: %s" % convergence_dict)
        return theta_opt, func_min


if __name__ == '__main__':
    from ase.io import read
    from gaussComparator import gaussComparator
    from featureCalculators.angular_fingerprintFeature_cy import Angular_Fingerprint
    from ase.visualize import view
    from ase import Atoms

    from custom_calculators import doubleLJ_calculator

    import matplotlib.pyplot as plt

    def finite_diff(krr, a, dx=1e-5, with_ud=False):
        pos0 = a.get_positions()
        Natoms, dim = pos0.shape
        F = np.zeros((Natoms, dim))
        vu = np.zeros((Natoms, dim))
        vd = np.zeros((Natoms, dim))
        for i in range(Natoms):
            for j in range(dim):
                pos_up = np.copy(pos0)
                pos_up[i,j] += dx/2
                pos_down = np.copy(pos0)
                pos_down[i,j] -= dx/2

                
                a_up = a.copy()
                a_down = a.copy()
                a_up.set_positions(pos_up)
                a_down.set_positions(pos_down)
                
                E_up, err_up, _ = krr.predict_energy(a_up, return_error=True)
                val_up = E_up - err_up
                E_down, err_down, _ = krr.predict_energy(a_down, return_error=True)
                val_down = E_down - err_down
                #print(E_down, E_up)
                #print(err_down, err_up)
                #print('val:', val_down - val_up)
                #print('E  :', E_down - E_up)

                vu[i,j] = val_up
                vd[i,j] = val_down
                
                F[i,j] = (val_down - val_up)/dx
        if with_ud:
            return F[0,0], vu[0,0], vd[0,0]
        else:
            return F
    
    def createData(r):
        positions = np.array([[0,0,0],[r,0,0]])
        a = Atoms('2H', positions, cell=[3,3,1], pbc=[0,0,0])
        calc = doubleLJ_calculator()
        a.set_calculator(calc)
        return a

    def test1():
        a_train = [createData(r) for r in [0.9,1,1.3,2,3]]
        
        #traj = read('graphene_data/all_every10th.traj', index='0::5')
        #a_train = traj[:100]
        E_train = np.array([a.get_potential_energy() for a in a_train])
        Natoms = len(a_train[0])
        #view(a_train)
        
        Rc1 = 5
        binwidth1 = 0.2
        sigma1 = 0.2
        
        Rc2 = 4
        Nbins2 = 30
        sigma2 = 0.2
        
        gamma = 1
        eta = 30
        use_angular = False
        
        featureCalculator = Angular_Fingerprint(a_train[0], Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
        
        
        # Set up KRR-model
        comparator = gaussComparator()
        krr = krr_class(comparator=comparator,
                        featureCalculator=featureCalculator)
        
        GSkwargs = {'reg': [1e-5], 'sigma': [5]}
        MAE, params = krr.train(atoms_list=a_train, data_values=E_train, k=3, add_new_data=False, **GSkwargs)
        print(MAE, params)
        
        Ntest = 100
        r_test = np.linspace(0.87, 3.5, Ntest)
        E_test = np.zeros(Ntest)
        err_test = np.zeros(Ntest)
        F_test = np.zeros(Ntest)
        E_true = np.zeros(Ntest)
        F_true = np.zeros(Ntest)
        F_num = np.zeros(Ntest)
        for i, r in enumerate(r_test):
            ai = createData(r)
            E, err, _ = krr.predict_energy(ai, return_error=True)
            E_test[i] = E
            err_test[i] = err
            
            F_test[i] = krr.predict_force(ai, with_error=True)[0]
            F_num[i] = finite_diff(krr, ai)[0,0]
            
            
            E_true[i] = ai.get_potential_energy()
            F_true[i] = ai.get_forces()[0,0]
            
            
        plt.figure()
        plt.plot(r_test, E_true, label='true')
        plt.plot(r_test, E_test, label='model')
        plt.plot(r_test, E_test-err_test, label='model')
        plt.legend()
        
        plt.figure()
        plt.plot(r_test, F_true, label='true')
        plt.plot(r_test, F_test, label='model')
        plt.plot(r_test, F_num, 'k:', label='num')
        plt.legend()

    
    def test2():
        traj = read('graphene_data/all_every10th.traj', index='0::5')
        a_train = traj[:100]
        E_train = np.array([a.get_potential_energy() for a in a_train])
        Natoms = len(a_train[0])

        Rc1 = 5
        binwidth1 = 0.2
        sigma1 = 0.2
        
        Rc2 = 4
        Nbins2 = 30
        sigma2 = 0.2
        
        gamma = 1
        eta = 30
        use_angular = False
        
        featureCalculator = Angular_Fingerprint(a_train[0], Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
        
        comparator = gaussComparator()
        krr = krr_class(comparator=comparator,
                        featureCalculator=featureCalculator)
        GSkwargs = {'reg': [1e-5], 'sigma': [5]}
        MAE, params = krr.train(atoms_list=a_train, data_values=E_train, k=3, add_new_data=False, **GSkwargs)
        print(MAE, params)
        
        a_test = traj[99]
        print(krr.predict_energy(a_test, return_error=True))
        
        F = krr.predict_force(a_test, with_error=True).reshape((-1,3))
        Fnum =  finite_diff(krr, a_test)
        print(F)
        print('')
        print(Fnum)
        print('')
        print((F - Fnum)/F)


    def test3():
        traj = read('graphene_data/all_every10th.traj', index='0::5')
        a_train = traj[:100]
        E_train = np.array([a.get_potential_energy() for a in a_train])
        Natoms = len(a_train[0])

        Rc1 = 5
        binwidth1 = 0.2
        sigma1 = 0.2
        
        Rc2 = 4
        Nbins2 = 30
        sigma2 = 0.2
        
        gamma = 1
        eta = 30
        use_angular = False
        
        featureCalculator = Angular_Fingerprint(a_train[0], Rc1=Rc1, Rc2=Rc2, binwidth1=binwidth1, Nbins2=Nbins2, sigma1=sigma1, sigma2=sigma2, gamma=gamma, eta=eta, use_angular=use_angular)
        
        comparator = gaussComparator()
        krr = krr_class(comparator=comparator,
                        featureCalculator=featureCalculator)
        GSkwargs = {'reg': [1e-5], 'sigma': [5]}
        MAE, params = krr.train(atoms_list=a_train, data_values=E_train, k=3, add_new_data=False, **GSkwargs)
        print(MAE, params)
        
        a_test = traj[99]
        
        def createData2(a0, r):
            a = a0.copy()
            positions = a.get_positions()
            positions[0,0] += r
            a.set_positions(positions)
            return a

        Ntest = 100
        r_test = np.linspace(-0.3, 0.3, Ntest)
        E_test = np.zeros(Ntest)
        err_test = np.zeros(Ntest)
        
        F_test = np.zeros(Ntest)
        F_num = np.zeros(Ntest)
        val_up = np.zeros(Ntest)
        val_down = np.zeros(Ntest)
        for i, r in enumerate(r_test):
            ai = createData2(a_test, r)
            E, err, _ = krr.predict_energy(ai, return_error=True)
            E_test[i] = E
            err_test[i] = err
            
            F_test[i] = krr.predict_force(ai, with_error=True)[0]
            #F_num[i] = finite_diff(krr, ai)[0,0]
            F_num[i], val_up[i], val_down[i]  = finite_diff(krr, ai, with_ud=True)
            
            
        plt.figure()
        plt.plot(r_test, E_test, label='model')
        plt.plot(r_test, E_test-err_test, label='val')
        plt.legend()

        plt.figure()
        val = E_test-err_test
        plt.plot(r_test, val_up-val, label='val up')
        plt.plot(r_test, val_down-val, label='val down')
        plt.legend()
        
        plt.figure()
        plt.plot(r_test, F_test, label='model')
        plt.plot(r_test, F_num, 'k:', label='num')
        plt.legend()

    def SOAP_test():
        pass

        
    test3()
    
    plt.show()
