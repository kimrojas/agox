import numpy as np
from os.path import join
from copy import copy
import pickle
import random

from numpy.random import default_rng
from itertools import product

from ase import Atoms
from ase.calculators.calculator import all_changes

from scipy.linalg import cholesky, cho_solve, solve_triangular, qr, lstsq, LinAlgError, svd, eig, eigvals, det
from scipy.spatial.distance import cdist
from scipy.optimize import fmin_l_bfgs_b

from agox.modules.models.model_ABC import ModelBaseClass
from time import time


class LSGPRModel(ModelBaseClass):
    name = 'LSGPRModel'
    implemented_properties = ['energy', 'forces', 'uncertainty']

    """
    Implements a sparse local Gaussian Process Model

    """
    def __init__(self, kernel=None, descriptor=None, noise=0.05, noise_bounds=(0.001, 0.05), episode_start_training=1,
                 update_interval=1, single_atom_energies=np.zeros(100), m_points=1000, adaptive_noise=False,
                 adaptive_noise_kwargs={'episodes': 5, 'factor':0.9, 'dE': 5., 'episode_start_updating': 5}, jitter=1e-8, force_method='forward',
                 transfer_data=[], sparsifier=None, method='lstsq', weights=None, prior=None, trainable_prior=False,
                 uncertainty_method='SR', **kwargs):
        
        super().__init__(**kwargs)

        # model parameters
        self.kernel = kernel    # this can be a standard sklearn kernel
        self.descriptor = descriptor         
        self._noise = noise      # this is in eV/atom
        
        self.noise_bounds = noise_bounds
        self.adaptive_noise = adaptive_noise
        self.adaptive_noise_kwargs = adaptive_noise_kwargs
        self.latest_model_errors = []
        
        self.force_method = force_method
        self.single_atom_energies = single_atom_energies
        self.transfer_data = transfer_data # add additional data not generated during run
        self.jitter = jitter    # to help inversion
        self.episode_start_training = episode_start_training
        self.update_interval = update_interval
        self.m_points = m_points
        self.sparsifier = sparsifier
        self.weights = weights
        # tmp:
        self.uncertainty_method = uncertainty_method
        self.prior = prior
        self.trainable_prior = trainable_prior
        
        if method in ['QR', 'cholesky', 'lstsq']:
            self.method = method
        else:
            print('Method not known - use: QR, cholesky or lstsq')
            print('Using lstsq as default.')
            self.method = 'lstsq'
        
        
        self.rng = default_rng()
        
        # Training info
        self.alpha = None
        self.Xn = None
        self.Xm = None
        self.m_indices = None
        self.y = None

        self.K_mm = None
        self.K_nm = None
        self.K_inv = None
        self.Kmm_inv = None
        self.L = None

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, s):
        if self.noise_bounds[0] <= s <= self.noise_bounds[1]:
            self._noise = s
        else:
            print('Trying to change noise above/below noise_bounds.')
        

    ####################################################################################################################
    # calculator stuff
    ####################################################################################################################
        
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if 'energy' in properties:
            e = self.predict_energy(atoms=atoms)
            self.results['energy'] = e
        
        if 'uncertainty' in properties:
            e_unc = self.predict_uncertainty(atoms=atoms)
            self.results['uncertainty'] = e_unc
            
        if 'forces' in properties:
            self.results['forces'] = self.predict_forces(atoms=atoms, method=self.force_method)

    ####################################################################################################################
    # Prediction
    ####################################################################################################################


    def predict_energy(self, atoms=None, X=None, return_uncertainty=False):
        if X is not None:
            pass
        elif X is None and atoms is not None:
            X = self.descriptor.get_local_environments(atoms)
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        if self.prior is not None:
            e0 = self.prior.predict_energy(atoms)
        else:
            e0 = 0

        k = self.kernel(self.Xm, X)
        if not return_uncertainty:
            return np.sum(k.T@self.alpha) + sum(self.single_atom_energies[atoms.get_atomic_numbers()]) + e0
        else:
            unc = self.predict_uncertainty(atoms=atoms, k=k)
            return np.sum(k.T@self.alpha) + sum(self.single_atom_energies[atoms.get_atomic_numbers()]) + e0, unc


    def predict_energies(self, atoms_list):
        energies = np.array([self.predict_energy(atoms) for atoms in atoms_list])
        return energies
            

    def predict_uncertainty(self, atoms=None, X=None, k=None):
        if X is not None:
            pass
        elif X is None and atoms is not None:
            X = self.descriptor.get_local_environments(atoms)
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        if k is None:
            k = self.kernel(self.Xm, X)

        if self.uncertainty_method == 'SR':
            unc = np.sqrt(np.sum(self.noise**2 * k.T @ self.K_inv @ k)) # sum variances then take square root
            return (len(atoms)-1)**2 * unc # assume local energies are completely correlated
        else:
            k0 = self.kernel(X,X)
            unc = np.sqrt(np.sum(k0) - np.sum(k.T@self.Kmm_inv@k) + np.sum(self.noise**2 * k.T @ self.K_inv @ k))
            return unc

        
    def predict_local_energy(self, atoms=None, X=None):
        """
        Calculate the local energies in the model. 
        """
        if X is not None:
            pass
        elif X is None and atoms is not None:
            X = self.descriptor.get_local_environments(atoms)            
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        k = self.kernel(self.Xm, X)
        return (k.T@self.alpha).reshape(-1,) + self.single_atom_energies[atoms.get_atomic_numbers()]



    def predict_forces(self, atoms=None, X=None, return_uncertainty=False, **kwargs):
        """
        method: central or forward, with forward being twice as fast
        """
        if return_uncertainty:  # always do forward for now
            f = self.predict_acquisition_forces_forward(atoms, **kwargs)
            unc = np.zeros(f.shape)
            return f, unc
            
        if self.force_method == 'forward':
            f = self.predict_forces_forward(atoms=atoms)
        else:
            f = self.predict_forces_central(atoms=atoms)
            
        return f


    def predict_forces_central(self, atoms=None, X=None):
        """
        Faster than the other method, but still numerical so not lightning speed.
        """
        # atoms = atoms_list[0].copy()

        d = 0.001

        all_atoms = []
        for a in range(len(atoms)):
            for i in range(3):
                patoms = atoms.copy()
                patoms.positions[a, i] += d
                matoms = atoms.copy()
                matoms.positions[a, i] -= d

                all_atoms.extend([patoms, matoms])

        energies = self.predict_energies(atoms_list=all_atoms)
        
        penergies = energies[0::2]
        menergies = energies[1::2]

        forces = ((menergies - penergies) / (2 * d)).reshape(len(atoms), 3)
        return forces

    
    def predict_force_central(self, atoms, index):
        d=0.001
        all_atoms = []
        for i in range(3):
            patoms = atoms.copy()
            patoms.positions[index, i] += d
            matoms = atoms.copy()
            matoms.positions[index, i] -= d
            all_atoms.extend([patoms, matoms])

        energies = self.predict_energies(atoms_list=all_atoms)
        
        penergies = energies[0::2]
        menergies = energies[1::2]

        force = ((menergies - penergies) / (2 * d)).reshape(1, 3)
        return force
    

    def predict_forces_forward(self, atoms=None, X=None):
        """
        Faster than the other method, but still numerical so not lightning speed.
        """
        # atoms = atoms_list[0].copy()

        d = 0.001

        all_atoms = [atoms.copy()]
        for a in range(len(atoms)):
            for i in range(3):
                patoms = atoms.copy()
                patoms.positions[a, i] += d

                all_atoms.extend([patoms])

        energies = self.predict_energies(atoms_list=all_atoms)

        penergies = energies[1::]
        ref_energies = np.ones(len(penergies))*energies[0]


        forces = ((ref_energies - penergies) / d).reshape(len(atoms), 3)
        return forces

    def predict_acquisition_forces_forward(self, atoms, acquisition_function=None):
        """
        Faster than the other method, but still numerical so not lightning speed.
        """
        # atoms = atoms_list[0].copy()

        d = 0.001

        all_atoms = [atoms.copy()]
        for a in range(len(atoms)):
            for i in range(3):
                patoms = atoms.copy()
                patoms.positions[a, i] += d

                all_atoms.extend([patoms])

        energies = []
        for a in all_atoms:
            e, std = self.predict_energy(a, return_uncertainty=True)
            energies.append(acquisition_function(e, std))

        penergies = energies[1::]
        ref_energies = np.ones(len(penergies))*energies[0]

        forces = ((ref_energies - penergies) / d).reshape(len(atoms), 3)
        return forces

    
    
    ####################################################################################################################
    # Training:
    # 3 levels:
    #     public: train_model
    #     changable in subclass: _train_sparse (sets: self.Xn, self.Xm, self.L, self.y)
    #         return: <boolean> training_nessesary
    #     private: _train_GPR (asserts that self.Xn, self.Xm, self.L, self.y is set)
    ####################################################################################################################


    def train_model(self, training_data, **kwargs):
        self.set_ready_state(True)
        self.atoms = None

        # train prior
        if self.prior is not None and self.trainable_prior:
            data = self.transfer_data + training_data
            energies = np.array([atoms.get_potential_energy() - sum(self.single_atom_energies[atoms.get_atomic_numbers()]) \
                                 for atoms in data])
            self.prior.train_model(data, energies)

        # prepare for training
        t1 = time()
        self.Xn, self.y = self._get_X_y(self.transfer_data + training_data)
        self._make_L(self.transfer_data + training_data)
        self._make_sigma(self.transfer_data + training_data)
        t2 = time()

        training_nessesary = self._train_sparse(atoms_list=self.transfer_data + training_data, **kwargs)
        if self.verbose:
            print('=========== MODEL INFO ===========')
            print(f'Number of energies available: {len(self.y)}')
            print(f'Number of local environments: {self.Xn.shape[0]}')
            print(f'Number of basis points:       {self.Xm.shape[0]}')
        if training_nessesary:
            t3 = time()
            self._train_GPR()
            t4 = time()
            if self.verbose:
                print('========== MODEL TIMING ==========')
                print(f'Total:          {t4-t1:.3f} s')
                print(f'Features:       {t2-t1:.3f} s')
                print(f'Sparsification: {t3-t2:.3f} s')
                print(f'Kernel:         {self.kernel_timing:.3f} s')
                print(f'Training:       {t4-t3-self.kernel_timing:.3f} s')


    def _train_sparse(self, atoms_list):
        """
        sparsification scheme: must set self.Xm
        returns boolean: indicates if training nessesary. 
        """
        if self.m_points > self.Xn.shape[0]:
            m_indices = np.arange(0,self.Xn.shape[0])
        else:
            m_indices = self.rng.choice(self.Xn.shape[0], size=self.m_points, replace=False)
        self.Xm = self.Xn[m_indices, :]
        return True
    
    def _train_GPR(self):
        """
        Fit Gaussian process regression model.
        Assert self.Xn, self.Xm, self.L and self.y is not assigned
        """
        assert self.Xn is not None, 'self.Xn must be set prior to call'
        assert self.Xm is not None, 'self.Xm must be set prior to call'
        assert self.L is not None, 'self.L must be set prior to call'
        assert self.y is not None, 'self.y must be set prior to call'

        t1 = time()
        self.K_mm = self.kernel(self.Xm)
        self.K_nm = self.kernel(self.Xn, self.Xm)
        t2 = time()
        self.kernel_timing = t2-t1
        
        LK_nm = self.L @ self.K_nm # This part actually also takes a long time to calculate - change in future
        K = self.K_mm + LK_nm.T @ self.sigma_inv @ LK_nm + self.jitter*np.eye(self.K_mm.shape[0])

        if self.uncertainty_method != 'SR':
            cho_Kmm = cholesky(self.K_mm+self.jitter*np.eye(self.K_mm.shape[0]), lower=True)
            self.Kmm_inv = cho_solve((cho_Kmm, True), np.eye(self.K_mm.shape[0]))

        if self.method == 'cholesky':
            t1 = time()
            K = self.nearestPD(K)
            t2 = time()
            cho_low = cholesky(K, lower=True)
            t3 = time()
            self.K_inv = cho_solve((cho_low, True), np.eye(K.shape[0]))
            t4 = time()
            self.alpha = cho_solve((cho_low, True), LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))
            t5 = time()
            
            if self.verbose > 1:
                print('========= FOR DEBUGGING ==========')
                print(f'PSD Kernel timing:             {t2-t1:.3f} s')
                print(f'Cholesky decomposition timing: {t3-t2:.3f} s')
                print(f'K inversion timing:            {t4-t3:.3f} s')
                print(f'alpha solve timing:            {t5-t4:.3f} s')
                print('')
                print(f'Cholesky residual:  {np.linalg.norm(cho_low.dot(cho_low.T) - K):.5f}')
                residual = np.linalg.norm((K @ self.K_inv)-np.eye(K.shape[0]))
                print(f'K inverse residual: {residual:.5f}')                
                residual = np.linalg.norm((K@self.alpha)-LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))
                print(f'alpha residual:     {residual:.5f}')

        elif self.method == 'QR':
            t1 = time()
            K = self.symmetrize(K)
            t2 = time()
            Q, R = qr(K)
            t3 = time()
            self.K_inv = solve_triangular(R, Q.T)
            t4 = time()
            self.alpha = solve_triangular(R, Q.T @ LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))
            t5 = time()
            
            if self.verbose > 1:
                print('========= FOR DEBUGGING ==========')
                print(f'Symmetrize Kernel timing:{t2-t1:.3f} s')
                print(f'QR decomposition timing: {t3-t2:.3f} s')
                print(f'K inversion timing:      {t4-t3:.3f} s')
                print(f'alpha solve timing:      {t5-t4:.3f} s')
                print('')
                print(f'QR residual: {np.linalg.norm(Q@R - K):.5f}')                
                residual = np.linalg.norm((K @ self.K_inv)-np.eye(K.shape[0]))
                print(f'K inverse residual: {residual:.5f}')                
                residual = np.linalg.norm((K@self.alpha)-LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))
                print(f'alpha residual:     {residual:.5f}')

        elif self.method == 'lstsq':
            t1 = time()
            K = self.symmetrize(K)
            t2 = time()
            Q, R = qr(K)
            t3 = time()
            self.K_inv = lstsq(R, Q.T)[0]
            t4 = time()
            self.alpha = lstsq(R, Q.T @ LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))[0]
            t5 = time()
            
            if self.verbose > 1:
                print('========= FOR DEBUGGING ==========')
                print(f'Symmetrize Kernel timing:{t2-t1:.3f} s')
                print(f'QR decomposition timing: {t3-t2:.3f} s')
                print(f'K inversion timing:      {t4-t3:.3f} s')
                print(f'alpha solve timing:      {t5-t4:.3f} s')
                print('')
                print(f'QR residual: {np.linalg.norm(Q@R - K):.5f}')                
                residual = np.linalg.norm((K @ self.K_inv)-np.eye(K.shape[0]))
                print(f'K inverse residual: {residual:.5f}')                
                residual = np.linalg.norm((K@self.alpha)-LK_nm.T @ self.sigma_inv @ self.y.reshape(-1,1))
                print(f'alpha residual:     {residual:.5f}')            

        else:
            print(f'method name: {self.method} unknown. Will fail shortly')
            
                        
    def update_model(self, new_data, all_data):
        t1 = time()
        if len(new_data) > 1:
            for data in new_data:
                self.Xn, self.y = self._update_X_y(data)
                self._update_L(new_data)
        else:
            self.Xn, self.y = self._update_X_y(new_data[0])
            self._update_L(new_data[0])

            
        self._make_sigma(self.transfer_data + all_data)
        t2 = time()

        training_nessesary = self._train_sparse(atoms_list=self.transfer_data + all_data)
        t3 = time()
        
        if self.verbose:
            print('=========== MODEL INFO ===========')
            print(f'Number of energies available: {len(self.y)}')
            print(f'Number of local environments: {self.Xn.shape[0]}')
            print(f'Number of basis points:       {self.Xm.shape[0]}')
            
        if training_nessesary:
            self._train_GPR()
            t4 = time()
            if self.verbose:
                print('========== MODEL TIMING ==========')
                print(f'Total:          {t4-t1:.3f} s')
                print(f'Features:       {t2-t1:.3f} s')
                print(f'Sparsification: {t3-t2:.3f} s')
                print(f'Kernel:         {self.kernel_timing:.3f} s')
                print(f'Training:       {t4-t3-self.kernel_timing:.3f} s')
        

                
    def neg_log_marginal_likelihood(self, theta):
        kernel = self.kernel.clone_with_theta(theta)
        K_nm = kernel(self.Xn, self.Xm)
        K_mm = kernel(self.Xm)
        K_mm = self.symmetrize(K_mm)
        Q, R = qr(K_mm)
        K_mm_inv = lstsq(R, Q.T)[0]
        l, e = eig(K_mm)
        print(l.shape, e.shape)
        l_tilde = self.Xn.shape[0]/self.Xm.shape[0] * l
        u_tilde = np.sqrt(self.Xn.shape[0]/self.Xm.shape[0]) * 1/l * K_nm @ e
        sigma_inv = np.eye(K_nm.shape[0])/self.noise**2
        print(K_mm.shape, sigma_inv.shape)
        inv = np.diag(l_tilde)/(u_tilde.T @ sigma_inv @ u_tilde)
        # Q, R = qr(np.diag(1/l_tilde) + u_tilde.T @ sigma_inv @ u_tilde)
        # inv = lstsq(R, Q.T)[0]
        print(inv.shape, u_tilde.shape)
        K_tilde_inv = sigma_inv - sigma_inv @ u_tilde @ inv @ u_tilde.T @ sigma_inv
        print(K_tilde_inv.shape, (u_tilde.T @ u_tilde).shape)
        (sign, log_det) = np.linalg.slogdet(np.eye(self.K_mm.shape[0])*self.noise**2 + l* (u_tilde.T @ u_tilde))
        # k_tilde = K_nm @ K_mm_inv @ K_nm.T
        
        # print('K_tilde shape:', K_tilde.shape)
        
        # K = K_mm + LK_nm.T @ self.sigma_inv @ LK_nm + self.jitter*np.eye(self.K_mm.shape[0])
        # K_tilde = self.symmetrize(K_tilde) + self.noise**2*np.eye(K_tilde.shape[0])
        
        # K_det = np.log(np.absolute(np.diagonal(R))).sum()
        # K_inv = lstsq(R, Q.T)[0]
        y = self.L.T @ self.y.reshape(-1,1)
        log_marg_like = -0.5 * y.T @ K_tilde_inv @ y - log_det - 0.5*len(self.y)*np.log(2*np.pi)
        print(30*'=')
        print(f'Hyper params: {np.exp(kernel.theta)}')
        print(f'neg_log_marg_like: {-log_marg_like}')
        print(30*'=')        
        return -log_marg_like


    def optimize_hyperparameters(self, maxiter=100):
        print('HYPERPARAMETER OPTIMIZATION NOT IMPLEMENTET!')
        likelihood = []
        thetas = []
        for theta in product(np.log([3,4,5,6,7,8]), np.log([10,15,20,25,30,35])):
            likelihood.append(self.neg_log_marginal_likelihood(theta))
            thetas.append(theta)
        theta_opt_idx = np.argmin(likelihood)
        theta_opt = thetas[theta_opt_idx]
        print('exp theta_opt:', np.exp(theta_opt))
        return theta_opt
        
        # theta_opt, func_min, convergence_dict = \
        #     fmin_l_bfgs_b(self.neg_log_marginal_likelihood,
        #                   self.kernel.theta,
        #                   bounds=self.kernel.bounds,
        #                   maxiter=maxiter,
        #                   pgtol=0.5,
        #                   approx_grad=True)
        # self.kernel._check_bounds_params()
        return theta_opt

                 
    ####################################################################################################################
    # Assignments:
    ####################################################################################################################

    def assign_from_main(self, main):
        super().assign_from_main(main)
        self.get_episode_counter = main.get_episode_counter

        if self.prior is not None:
            self.prior.assign_from_main(main)
        
        if not self.part_of_delta_model:
            main.database.attach_observer(self.name, self.training_observer_func)


    def training_observer_func(self, database):
        episode = self.get_episode_counter()

        if episode < self.episode_start_training:
            return
        if (episode % self.update_interval != 0) * (episode != self.episode_start_training):
            return


        all_data = database.get_all_candidates()
        
        if self.adaptive_noise:
            self._update_noise(all_data, **self.adaptive_noise_kwargs)

        if self.sparsifier is not None:
            full_update, data_for_training = self.sparsifier(all_data)
        elif self.ready_state:
            full_update = False
            data_amount_before = len(self.y) - len(self.transfer_data)            
            data_for_training = all_data
            data_amount_new = len(data_for_training) - data_amount_before
            new_data = data_for_training[-data_amount_new:]            
        else:
            full_update = True
            data_for_training = all_data

        if full_update:
            self.train_model(data_for_training)
        else:
            self.update_model(new_data, data_for_training)
        

        
        
    ####################################################################################################################
    # Helpers
    ####################################################################################################################

    def _update_noise(self, all_data, episodes=5, factor=0.9, dE=5, episode_start_updating=5):
        if not self.ready_state:
            return
        e = all_data[-1].get_potential_energy()
        p = self.predict_energy(all_data[-1])
        self.latest_model_errors.append(e-p)
        self.latest_model_errors[-episodes:]
        
        if self.get_episode_counter() < episode_start_updating:
            return
        
        mae = np.mean(np.absolute(self.latest_model_errors))
        
        print(5*'-', 'ADAPTIVE NOISE', 5*'-')
        print(f'Noise before: {self.noise}')
        
        if mae > dE:
            self.noise = self.noise_bounds[1] + (self.noise - self.noise_bounds[1]) * factor
        else:
            self.noise = self.noise_bounds[0] + (self.noise - self.noise_bounds[0]) * factor
            
        print(f'Noise after: {self.noise}')
        print(30*'-')

            

    def _get_X_y(self, atoms_list):

        X = self.descriptor.get_local_environments(atoms_list)
        y = np.array([atoms.get_potential_energy() - sum(self.single_atom_energies[atoms.get_atomic_numbers()]) \
                      for atoms in atoms_list])
        if self.prior is not None:
            y -= np.array([self.prior.predict_energy(atoms) for atoms in atoms_list])
            
        return X, y

    def _update_X_y(self, new_atoms):
        new_x = self.descriptor.get_local_environments(new_atoms)
        X = np.vstack((self.Xn, new_x))
        new_y = new_atoms.get_potential_energy() - sum(self.single_atom_energies[new_atoms.get_atomic_numbers()])
        y = np.append(self.y, new_y)
        return X, y
    
    def _make_L(self, atoms_list):
        assert self.y is not None, 'self.y cannot be None'
        assert self.Xn is not None, 'self.Xn cannot be None'
        col = 0
        self.L = np.zeros((len(self.y), self.Xn.shape[0]))
        for i, atoms in enumerate(atoms_list):
            self.L[i,col:col+len(atoms)] = 1.
            col += len(atoms)
            
    def _update_L(self, new_atoms):
        new_col = np.zeros((self.L.shape[0], len(new_atoms)))
        self.L = np.hstack((self.L, new_col))
        new_row = np.zeros((1, self.L.shape[1]))
        new_row[0, -len(new_atoms):] = 1.
        self.L = np.vstack((self.L, new_row))

    def _make_sigma(self, atoms_list):
        self.sigma_inv = np.diag([1/(len(atoms)*self.noise**2) for atoms in atoms_list])
        if self.weights is not None:
            print(self.sigma_inv.shape, self.weights.shape)
            self.sigma_inv[np.diag_indices_from(self.sigma_inv)] *= self.weights
        
    def symmetrize(self, A):
        return (A + A.T)/2
    
    def nearestPD(self, A):
        """Find the nearest positive-definite matrix to input
        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].
        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
        """
        B = self.symmetrize(A)
        if self.isPD(B):
            return B

        _, s, V = svd(B)
        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        if self.isPD(A3):
            return A3
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self.isPD(A3):
            mineig = np.min(np.real(eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        return A3    
        
    def isPD(self, A):
        try:
            _ = cholesky(A, lower=True)
            return True
        except LinAlgError:
            return False            

    def get_model_parameters(self):
        parameters = {}
        parameters['Xm'] = self.Xm
        parameters['K_inv'] = self.K_inv
        parameters['Kmm_inv'] = self.Kmm_inv
        parameters['alpha'] = self.alpha
        parameters['single_atom_energies'] = self.single_atom_energies
        parameters['theta'] = self.kernel.theta
        return parameters

    def set_model_parameters(self, parameters):
        self.Xm = parameters['Xm']
        self.K_inv = parameters['K_inv']
        self.Kmm_inv = parameters['Kmm_inv']
        self.alpha = parameters['alpha']
        self.single_atom_energies = parameters['single_atom_energies']
        self.kernel.theta = parameters['theta']
        self.set_ready_state(True)

    
    def save(self, directory='', prefix='my-model'):
        with open(join(directory, prefix+'.lsgpr'), 'wb') as handle:
            pickle.dump(self, handle)
            
    @classmethod
    def load(self, path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)


    

