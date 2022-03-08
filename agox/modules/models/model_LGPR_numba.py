import numpy as np
from os.path import join
from copy import copy
import pickle

from ase import Atoms
from ase.calculators.calculator import all_changes

from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
from scipy.optimize import fmin_l_bfgs_b

from .model_ABC import ModelBaseClass


class LocalGPRModel(ModelBaseClass):
    name = 'localGPR'
    implemented_properties = ['energy', 'forces', 'uncertainty']

    """
    Implements a local Gaussian Process Model

    The key feature of such a model is that the covariance-matrix is a sum of local covariances (or kernel matrices.)
    """
    def __init__(self, kernel, descriptor, noise=3e-3, sparsifier=None, episode_start_training=1,
                 update_interval=1, optimize_interval=20, single_atom_energies=np.zeros(100), sparse=False,
                 use_bias=False, jitter=10e-8, force_method='central', transfer_data=[], max_variance=25, **kwargs):
        super().__init__(**kwargs)

        # model parameters
        self.kernel_ = kernel
        self.noise = noise
        self.descriptor = descriptor
        self.sparsifier = sparsifier
        self.force_method = force_method

        self.sparse = sparse
        self.jitter = jitter
        
        # Training Arguments:
        self.episode_start_training = episode_start_training
        self.update_interval = update_interval
        self.optimize_interval = optimize_interval

        # Training info
        self.max_variance = max_variance
        self.alpha = None
        self.X_train_ = None
        self.y_train_ = None

        if self.sparse:
            self.alpha_sparse = None
            self.X_train_sparse = None
            self.y_train_sparse = None
        

        self.single_atom_energies = single_atom_energies
        self.use_bias = use_bias
        self.bias_per_atom = 0

        # Transfer info
        self.number_of_transfer_strucs = 0
        self.transfer_data = transfer_data



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
            self.results['forces'] = self.predict_forces(atoms=atoms)# , method=self.force_method)

    ####################################################################################################################
    # Prediction
    ####################################################################################################################


    def predict_energy(self, atoms=None, X=None, return_X=False):
        if X is not None:
            pass
        elif X is None and atoms is not None:
            X = np.expand_dims(self.descriptor(atoms), axis=0)
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        if self.sparse:
            k = self.kernel_(X, self.Xm_train)
            y_mean = k.dot(self.alpha_sparse)[0]
        else:
            k = self.kernel_(X, self.X_train_)
            y_mean = k.dot(self.alpha)[0]
        if atoms is not None:
            y_mean += sum(self.single_atom_energies[atoms.get_atomic_numbers()]) + self.bias_per_atom*len(atoms)

        if return_X:
            return y_mean, X
        else:
            return y_mean


    def predict_uncertainty(self, atoms=None, X=None):
        if X is not None:
            pass
        elif X is None and atoms is not None:
            X = np.expand_dims(self.descriptor(atoms), axis=0)
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')
            
        if self.sparse:
            k = self.kernel_(X, self.Xm_train)
            unc = np.sqrt(np.sum(self.noise * k @ self._K_inv_sparse @ k.T))
        else:
            k0 = self.kernel_(X)
            k = self.kernel_(X, self.X_train_)
            vk = np.dot(self._K_inv,k.reshape(-1))
            unc = np.sqrt(np.sum(k0 - np.dot(k,vk)))
        return unc

        
    def predict_energies(self, atoms_list=None, X=None, return_error=False):
        if X is not None:
            pass
        elif X is None and atoms_list is not None:
            X = np.array([self.descriptor(atoms) for atoms in atoms_list])
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        if self.sparse:
            k = self.kernel_(X, self.Xm_train)
            y_mean = k.dot(self.alpha_sparse)
        else:
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha)

        if atoms_list is not None:
            single_atom_corrections = np.array([sum(self.single_atom_energies[atoms.get_atomic_numbers()]) + self.bias_per_atom*len(atoms) for atoms in atoms_list])
            y_mean += single_atom_corrections
        
        if return_error:
            E_unc = self.predict_uncertainty(X=X)
            return y_mean, E_unc, K_trans
        else:
            return y_mean
            

    def predict_local_energies(self, X=None, atoms_list=None, return_unc=False):
        """
        Calculate the local energies in the model. 

        Efficiency not gauranteed.
        """
        if X is not None:
            pass
        elif X is None and atoms_list is not None:
            X = np.array([self.descriptor(atoms) for atoms in atoms_list])
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        y_mean = []
        y_mean_unc = []
        for x in X:
            atoms_shape = x.shape[0]
            e_local = np.zeros(atoms_shape)
            for l in range(atoms_shape):
                for n in range(self.X_train_.shape[0]):
                    for j in range(self.X_train_[n].shape[0]):
                        k = self.kernel_.get_local_kernel(x[l].reshape(1,-1), self.X_train_[n][j].reshape(1,-1))
                        e_local[l] += self.alpha[n] * np.sum(k)
            y_mean.append(e_local)
            
            if return_unc:
                e_local_unc = np.zeros(atoms_shape)
                for l in range(atoms_shape):
                    k0 = self.kernel_.get_local_kernel(x[l].reshape(1,-1),x[l].reshape(1,-1))
                    k = np.zeros(self.X_train_.shape[0])
                    for j in range(self.X_train_.shape[0]):
                        for i in range(self.X_train_.shape[1]):
                            k[j] += self.kernel_.get_local_kernel(x[l].reshape(1,-1),self.X_train_[j][i].reshape(1,-1))
                    vk = np.dot(self._K_inv,k.reshape(-1))

                    e_local_unc[l] = np.sqrt(k0 - np.dot(k,vk))
                y_mean_unc.append(e_local_unc)
                
        if return_unc:
            return y_mean, y_mean_unc
        else:
            return y_mean



    def predict_forces(self, atoms=None, X=None, method='central'):
        """
        method: central or forward, with forward being twice as fast
        """

        if method == 'forward':
            f = self.predict_forces_forward(atoms=atoms)
        else:
            f = self.predict_forces_central(atoms=atoms)
            
        return f


    def predict_forces_central(self, X=None, atoms=None):
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

        # print(len(all_atoms))
        # X = np.array([self.descriptor(atoms) for atoms in all_atoms])

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
    

    def predict_forces_forward(self, X=None, atoms=None):
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

        # X = np.array([self.descriptor(atoms) for atoms in all_atoms])

        energies = self.predict_energies(atoms_list=all_atoms)

        penergies = energies[1::]
        ref_energies = np.ones(len(penergies))*energies[0]


        forces = ((ref_energies - penergies) / d).reshape(len(atoms), 3)
        return forces

    
    def batch_predict(self, data, over_write=False):
        energies = self.predict_energies(atoms_list=data)

        if over_write:
            self.batch_assign(data, energies)
        return energies

    
    ####################################################################################################################
    # Training
    ####################################################################################################################

    def _train_GPR(self, X=None, y=None, atoms_list=None):
        """Fit Gaussian process regression model.
        """

        if X is not None:
            assert y is not None
        elif X is None and atoms_list is not None:
            
            if self.sparsifier is not None:
                atoms_list = self.sparsifier(copy(atoms_list))
                
            X = np.array([self.descriptor(atoms) for atoms in atoms_list])
            if y is None:
                y = np.array([atoms.get_potential_energy() - sum(self.single_atom_energies[atoms.get_atomic_numbers()]) \
                              for atoms in atoms_list])
                num_atoms = np.array([len(atoms) for atoms in atoms_list])
                if self.use_bias:
                    self.bias_per_atom = np.mean(y/num_atoms)

                y = np.array([y[i]-num_atoms[i]*self.bias_per_atom for i in range(len(y))])
                
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        self.X_train_ = np.copy(X) #if self.copy_X_train else X
        self.y_train_ = np.copy(y) #if self.copy_X_train else y

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.noise
        self.L_ = cholesky(K, lower=True)  # Line 2
        self._K_inv = cho_solve((self.L_, True), np.eye(K.shape[0]))
        self.alpha = self._K_inv @ self.y_train_ # cho_solve((self.L_, True), self.y_train_)  # Line 3
        
        return self.alpha


    def _train_sparse_GPR(self, X=None, y=None, atoms_list=None):
        """Fit sparse Gaussian process regression model.
        """

        if X is not None:
            assert y is not None
        elif X is None and atoms_list is not None:
            
            if self.sparsifier is not None:
                sparse_atoms_list = self.sparsifier(copy(atoms_list))
                print(len(atoms_list), len(sparse_atoms_list)) # I probably have to copy
                
            Xm = np.array([self.descriptor(atoms) for atoms in sparse_atoms_list])
            Xn = np.array([self.descriptor(atoms) for atoms in atoms_list])
            if y is None:
                yn = np.array([atoms.get_potential_energy() - sum(self.single_atom_energies[atoms.get_atomic_numbers()]) \
                              for atoms in atoms_list])
                num_atoms = np.array([len(atoms) for atoms in atoms_list])
                if self.use_bias:
                    self.bias_per_atom = np.mean(yn/num_atoms)

                yn = np.array([yn[i]-num_atoms[i]*self.bias_per_atom for i in range(len(yn))])
                
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        self.Xm_train = np.copy(Xm) #if self.copy_X_train else X
        self.Xn_train = np.copy(Xn) #if self.copy_X_train else X
        self.yn_train = np.copy(yn) #if self.copy_X_train else y

        Kmm = self.kernel_(self.Xm_train)
        Knm = self.kernel_(self.Xn_train, self.Xm_train)
        
        Kmm[np.diag_indices_from(Kmm)] += self.jitter # to help inversion

        K = np.matmul(Knm.T, Knm) + self.noise*Kmm
        self.L_sparse = cholesky(K, lower=True)
        self._K_inv_sparse = cho_solve((self.L_sparse, True), np.eye(K.shape[0]))

        self.alpha_sparse = self._K_inv_sparse @ (Knm.T @ self.yn_train)
        
        return self.alpha_sparse

    
    def train_model(self, training_data):
        self.set_ready_state(True)
        # Wipe any stored atoms object, since now the model will change and hence give another result
        self.atoms = None
        if self.sparse:
            self._train_sparse_GPR(atoms_list=training_data)
        else:
            self._train_GPR(atoms_list=training_data)


    def update_model(self, new_data):
        if isinstance(new_data, list):
            X_new = np.array([self.descriptor(atoms) for atoms in new_data])
            y_new = np.array([atoms.get_potential_energy() - sum(self.single_atom_energies[atoms.get_atomic_numbers()]) \
                              for atoms in new_data])
        elif isinstance(new_data, Atoms):
            X_new = np.expand_dims(self.descriptor(new_data), axis=0)
            y_new = np.array([new_data.get_potential_energy() - sum(self.single_atom_energies[new_data.get_atomic_numbers()])])
        else:
            pass

        if not self.use_bias:
            y = np.concatenate((self.y_train_,y_new))
        else:
            print('update with bias not implementet yet - model will be wrong!')
        
        if len(self.X_train_.shape) == 1 and len(X_new.shape) == 1: # different number of atoms in training and new data
            X = np.array([x for x in self.X_train_] + [x for x in X_new])
        elif len(self.X_train_.shape) == 3 and len(X_new.shape) == 1: # same number of atoms in training and different in new data
            X = np.array([self.X_train_[i] for i in range(self.X_train_.shape[0])] + [x for x in X_new])
        elif len(self.X_train_.shape) == 1 and len(X_new.shape) == 3: # different number of atoms in training and same in new data
            X = np.array([x for x in self.X_train_] + [X_new[i] for i in range(X_new.shape[0])])
        elif len(self.X_train_.shape) == 3 and len(X_new.shape) == 3: # same number of atoms in training and in new data
            X = np.vstack((self.X_train_, X_new))


        self._train_GPR(X=X, y=y)



    # def sparsify_data(self, atoms_list):
    #     sparse_list = atoms_list
    #     y = np.array([atoms.get_potential_energy() - \
    #                   sum(self.single_atom_energies[atoms.get_atomic_numbers()]) for atoms in sparse_list])
    #     p = (y-np.amin(y))/(np.amax(y)-np.amin(y))
    #     p /= p.sum()
    #     while np.var(y) > self.max_variance:
    #         index_remove = np.random.choice(len(y), p=p) # np.argmax(y)
    #         sparse_list.pop(index_remove)
    #         y = np.delete(y, index_remove)
    #         p = (y-np.amin(y))/(np.amax(y)-np.amin(y))
    #         p /= p.sum()
            
    #     return sparse_list
            
        

    ####################################################################################################################
    # Hyperparameter optimization
    ####################################################################################################################
    
    def neg_log_marginal_likelihood(self, hyperparameters):
        # get existing hyperparameter
        hyperparameters_now = copy(self.kernel_.hyperparameters)

        # set new hyperparameter
        self.kernel_.hyperparameters = hyperparameters

        assert self.X_train_ is not None, 'self.X_train_ is None'
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.noise

        L_ = cholesky(K, lower=True)
        K_inv = cho_solve((L_, True), np.eye(K.shape[0]))
        alpha = K_inv @ self.y_train_

        log_marg_like = -0.5 * np.dot(self.y_train_,alpha) - np.log(np.diag(L_)).sum() - 0.5*len(self.y_train_)*np.log(2*np.pi)

        jac_log_marg_like = np.zeros(hyperparameters.shape)
        t1 = np.einsum("i,j->ij", alpha, alpha) - K_inv
        t2 = self.kernel_.get_kernel_gradient(self.X_train_)
        for i in range(len(hyperparameters)):
            jac_log_marg_like[i] = 0.5 * np.einsum('ij,ji->', t1,t2[i])
        
        # change parameters back
        self.kernel_.hyperparameters = hyperparameters_now
        print('neg_log_marg_likelihood called', -log_marg_like, -jac_log_marg_like, hyperparameters)
        return -log_marg_like, -jac_log_marg_like


    def optimize_hyperparameters(self, maxiter=100):
        # A = np.sqrt(np.var(self.y_train_))
        # self.kernel_.A = A
        theta_initial = self.kernel_.hyperparameters
        theta_bounds = self.kernel_.hyperparameter_bounds
        theta_opt, func_min, convergence_dict = \
            fmin_l_bfgs_b(self.neg_log_marginal_likelihood,
                          theta_initial,
                          bounds=theta_bounds,
                          maxiter=maxiter,
                          pgtol=0.5)
        self.kernel_.hyperparameters = theta_opt
        return theta_opt
                 
    ####################################################################################################################
    # Assignments:
    ####################################################################################################################

    def assign_from_main(self, main):
        super().assign_from_main(main)
        self.get_episode_counter = main.get_episode_counter
        
        if not self.part_of_delta_model:
            main.database.attach_observer(self.name, self.training_observer_func)


    def training_observer_func(self, database):
        episode = self.get_episode_counter()
        if episode < self.episode_start_training:
            return
        if (episode % self.update_interval != 0) * (episode != self.episode_start_training):
            return

        if episode % self.optimize_interval == 0 and episode > self.episode_start_training:
            self.optimize_hyperparameters()

        all_data = database.get_all_candidates()
        if self._ready_state and isinstance(all_data, list) and len(all_data)<self.max_training_data:
            # data = self.sparsify_data(all_data)
            self.train_model(all_data + self.transfer_data)
            # new_data = all_data[(len(self.y_train_)-self.number_of_transfer_strucs):]
            # print('updating model!!')
            # self.update_model(new_data)

        
        
    ####################################################################################################################
    # Misc.
    ####################################################################################################################

    def save_model(self, directory='', prefix=''):
        save_dict = {
            'alpha': self.alpha,
            'X_train': self.X_train_,
            'y_train': self.y_train_,
            'num_transfer': self.number_of_transfer_strucs
        }
        with open(join(directory, prefix+'model.pkl'), 'wb') as handle:
            pickle.dump(save_dict, handle)
            
    
    def load_model(self, directory='', prefix='', transfer=False):
        """
        Loads saved model weights and features, but does not check that you are using the 
        same kernel or hyperparameters! 
        """
        with open(join(directory, prefix+'model.pkl'), 'rb') as handle:
            load_dict = pickle.load(handle)

        self.alpha = load_dict['alpha']
        self.X_train_ = load_dict['X_train']
        self.y_train_ = load_dict['y_train']
        self.number_of_transfer_strucs = load_dict['num_transfer']
        if transfer:
            self.number_of_transfer_strucs += len(self.y_train_)


    
