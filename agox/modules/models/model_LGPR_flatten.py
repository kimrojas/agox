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

class FlattenedLGPRModel(ModelBaseClass):
    name = 'localGPRflattened'
    implemented_properties = ['energy', 'forces', 'uncertainty']

    """
    Implements a local Gaussian Process Model

    The key feature of such a model is that the covariance-matrix is a sum of local covariances (or kernel matrices.)
    """
    def __init__(self, kernel, noise=3e-3, descriptor=None, episode_start_training=1, update_interval=1, optimize_interval=20,
                 max_training_data=1E10, single_atom_energies=np.zeros(100), use_bias=False, num_best=0, force_method='central',
                 transfer_data=[], max_variance=25, full_LGPR_calculator=None, number_of_best_local_environments = {},
                 number_of_random_local_environments = None, number_of_different_local_environments = {}, cluster_distance=4,
                 full_structures_max=10000, full_structures_best=10000, **kwargs):
        super().__init__(**kwargs)

        # model parameters
        self.kernel_ = kernel
        self.noise = noise
        self.descriptor = descriptor
        self.force_method = force_method

        # Training Arguments:
        self.episode_start_training = episode_start_training
        self.max_training_data = max_training_data
        self.update_interval = update_interval
        self.optimize_interval = optimize_interval
        
        self.num_best = num_best
        self.num_random = max_training_data - num_best

        # Training info
        self.max_variance = max_variance
        self.alpha = None
        self.X_train_ = None
        self.y_train_ = None

        self.single_atom_energies = single_atom_energies
        self.use_bias = use_bias
        self.bias_per_atom = 0

        # Transfer info
        self.number_of_transfer_strucs = 0
        self.transfer_data = transfer_data

        self.full_LGPR_calculator = full_LGPR_calculator
        self.number_of_best_local_environments = number_of_best_local_environments
        if number_of_random_local_environments is None:
            self.number_of_random_local_environments = number_of_best_local_environments
        else:
            self.number_of_random_local_environments = number_of_random_local_environments
        self.number_of_different_local_environments = number_of_different_local_environments
        self.cluster_distance = cluster_distance

        self.full_structures_max = full_structures_max
        self.full_structures_best = full_structures_best


    ####################################################################################################################
    # calculator stuff
    ####################################################################################################################
        
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if 'energy' in properties:
            e_tot = 0
            struc_descriptor = self.descriptor(atoms)
            e_tot = self.predict_energy(atoms,X=np.expand_dims(struc_descriptor, axis=1))
            #print('{:8.3f}'.format(e_tot))
            self.results['energy'] = e_tot
        
        if 'uncertainty' in properties:
            e_unc = 0
            #e_unc = self.predict_uncertainty(atoms=atoms)
            self.results['uncertainty'] = e_unc
            
        if 'forces' in properties:
            self.results['forces'] = self.predict_forces(atoms=atoms, method=self.force_method)

    ####################################################################################################################
    # Prediction
    ####################################################################################################################


    def predict_energy(self, atoms=None, X=None, return_X=False):
        if X is not None:
            pass
        elif X is None and atoms is not None:
            pass
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        K_trans = self.kernel_(X, self.X_train_)
        y_mean = sum(K_trans.dot(self.alpha))
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

        k0 = self.kernel_(X)
        k = self.kernel_(X, self.X_train_)
        vk = np.dot(self._K_inv,k.reshape(-1))
        return np.sqrt(np.sum(k0 - np.dot(k,vk)))

        
    def predict_energies(self, atoms_list=None, X=None, return_error=False):
        if X is not None:
            pass
        elif X is None and atoms_list is not None:
            X = np.array([self.descriptor(atoms) for atoms in atoms_list])
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        Xdim = X.shape
        Nstrucs = Xdim[0]
        Natoms = Xdim[1]
        Nsoap = Xdim[2]
        X = X.reshape((Nstrucs * Natoms, 1, Nsoap))
        K_trans = self.kernel_(X, self.X_train_)
        #print(X.shape,self.X_train_.shape,K_trans.shape,self.alpha.shape)
        y = K_trans.dot(self.alpha)
        y = y.reshape((Nstrucs,Natoms))
        y_mean = np.sum(y, axis=1)

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

    def train_GPR(self, X=None, y=None, atoms_list=None):
        """Fit Gaussian process regression model.

        Parameters
        ----------
        X : sequence of length n_samples
            Feature vectors or other representations of training data.
            Could either be array-like with shape = (n_samples, n_features)
            or a list of objects.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values

        Returns
        -------
        self : returns an instance of self.
        """

        if atoms_list is not None and len(atoms_list) > self.max_training_data:
            selected = np.zeros(self.max_training_data).astype(int)
            sort_indexs = np.argsort(y)
            selected[0:self.num_best] = sort_indexs[0:self.num_best] # Take num_best of the lowest energy structures.
            # Randomly take some of the remaining structures. 
            selected[self.num_best:] = np.random.choice(sort_indexs[self.num_best::], size=self.num_random, replace=False) 
            atoms_list = [atoms_list[i] for i in selected]
            y = y[selected]            

        if X is not None:
            assert y is not None
        elif X is None and atoms_list is not None:
            X = np.array([self.descriptor(atoms) for atoms in atoms_list])
            if y is None:
                # y = [atoms.get_potential_energy() for atoms in atoms_list]
                y = np.array([atoms.get_potential_energy() - sum(self.single_atom_energies[atoms.get_atomic_numbers()]) for atoms in atoms_list])
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
        print('TRAINING',self.X_train_.shape)
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.noise
        self.L_ = cholesky(K, lower=True)  # Line 2
        self._K_inv = cho_solve((self.L_, True), np.eye(K.shape[0]))
        self.alpha = self._K_inv @ self.y_train_ # cho_solve((self.L_, True), self.y_train_)  # Line 3
        
        return self.alpha

    def train_model(self, training_data):
        self.set_ready_state(True)
        # Wipe any stored atoms object, since now the model will change and hence give another result
        self.atoms = None

        # train model with kernel where every element is a double sum over atoms in each structure
        self.full_LGPR_calculator.train_model(training_data)

        # establish local environments and corresponding local energies
        local_environments = []
        local_energies = []
        local_atomic_numbers = []
        for j, struc in enumerate(training_data):
            #print('=========================')
            #print('==========',j,'============')
            #print('=========================')
            verbose = j == 0 or j > len(training_data) - 20
            debug = False
            if verbose and debug:
                e_struc_wise = struc.get_potential_energy()
            e_atom_wise = 0
            struc_descriptor = self.descriptor(struc)
            e_atoms = self.full_LGPR_calculator.predict_energies(X=np.expand_dims(struc_descriptor, axis=1))
            if verbose and debug:
                e_struc = self.full_LGPR_calculator.predict_energy(X=np.expand_dims(struc_descriptor, axis=0))
            for i,atom in enumerate(struc):
                e_atom = e_atoms[i]
                local_environment = struc_descriptor[i:i+1]
                e_atom_wise += e_atom
                local_environments.append(local_environment)
                #print('ENV','[' + ','.join(['{:5.2f}'.format(e) for e in local_environment[0]]) + ']')
                local_energies.append(e_atom)
                local_atomic_numbers.append(atom.number)
                if verbose:
                    print('{:8.3f}'.format(e_atom),{1: 'H', 6: 'C', 7: 'N', 8: 'O'}.get(atom.number,'NA'),end='')
            if verbose:
                if debug:
                    print(' {:8.3f} {:8.3f} {:8.3f}'.format(e_struc_wise,e_atom_wise,e_struc))
                else:
                    print(' {:8.3f} '.format(e_atom_wise))

        #print('........................')
        local_environments = np.array(local_environments)
        local_energies = np.array(local_energies)
        local_atomic_numbers = np.array(local_atomic_numbers)

        local_env_2_dim = local_environments.reshape((local_environments.shape[0],local_environments.shape[2]))

        # skip sparcification if not specified
        if list(self.number_of_best_local_environments.keys()) == [] and \
           list(self.number_of_different_local_environments.keys()) == []:
            self.train_GPR(X=local_environments,y=local_energies)
            return

        # do sparcification I
        #
        # pick best local envs and random local envs
        selected_indices_atomic_number_wise = {}
        for atomic_number in self.number_of_best_local_environments.keys():
            indices_unsorted = np.array(range(len(local_atomic_numbers)))[local_atomic_numbers == atomic_number]
            indices_sorted = sorted(indices_unsorted,key=lambda x: local_energies[x])
            len_best = min(self.number_of_best_local_environments[atomic_number],round(len(indices_sorted)/2))
            len_random = min(self.number_of_random_local_environments[atomic_number],len(indices_sorted) - len_best)
            selected_indices_atomic_number_wise[atomic_number] = list(indices_sorted[:len_best]) \
                                                                 + list(np.random.choice(indices_sorted[len_best:], size=len_random, replace=False))
            print('SPARSIFICATION1:',atomic_number,len_best,len_random)

        # skip sparcification if not specified
        if list(self.number_of_different_local_environments.keys()) == []:
            selected_indices = []
            for atomic_number in selected_indices_atomic_number_wise.keys():
                selected_indices += selected_indices_atomic_number_wise[atomic_number]
            self.train_GPR(X=local_environments[selected_indices],y=local_energies[selected_indices])
            return

            
        # do sparcification II
        
        selected_indices = []
        for atomic_number in self.number_of_different_local_environments.keys():
            indices = selected_indices_atomic_number_wise[atomic_number]

            sample_indices = [indices[0]]
            for i in indices[1:]:
                r = cdist(local_env_2_dim[i].reshape(1,-1),local_env_2_dim[sample_indices])
                d_min = np.min(r)
                if d_min > self.cluster_distance:
                    sample_indices.append(i)
                    if len(sample_indices) >= self.number_of_different_local_environments[atomic_number]:
                        break
            selected_indices += sample_indices
            print('SPARSIFICATION2:',atomic_number,len(sample_indices))

        self.train_GPR(X=local_environments[selected_indices],y=local_energies[selected_indices])

    def update_model(self, new_data):
        if isinstance(new_data, list):
            X_new = np.array([self.descriptor(atoms) for atoms in new_data])
            y_new = np.array([atoms.get_potential_energy() - sum(self.single_atom_energies[atoms.get_atomic_numbers()]) for atoms in new_data])
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


        self.train_GPR(X=X, y=y)



    def sparsify_data(self, atoms_list):
        sparse_list = atoms_list
        y = np.array([atoms.get_potential_energy() - \
                      sum(self.single_atom_energies[atoms.get_atomic_numbers()]) for atoms in sparse_list])
        p = (y-np.amin(y))/(np.amax(y)-np.amin(y))
        p /= p.sum()
        while np.var(y) > self.max_variance:
            index_remove = np.random.choice(len(y), p=p) # np.argmax(y)
            sparse_list.pop(index_remove)
            y = np.delete(y, index_remove)
            p = (y-np.amin(y))/(np.amax(y)-np.amin(y))
            p /= p.sum()
            
        return sparse_list
            
        

    ####################################################################################################################
    # Hyperparameter optimization
    ####################################################################################################################
    
    def neg_log_marginal_likelihood(self, hyperparameters):
        # get existing hyperparameter
        hyperparameters_now = copy(self.kernel_.hyperparameters)

        # set new hyperparameter
        self.kernel_.hyperparameters = hyperparameters

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

        all_data = database.get_all_candidates()
        if episode % self.optimize_interval == 0 and episode > self.episode_start_training and len(all_data) > 2:
            self.optimize_hyperparameters()
            print(f'Hyperparameters optimized to: A={self.kernel_.A:.3f}, ls={self.kernel_.ls:.3f}')

        if self._ready_state and isinstance(all_data, list) and len(all_data) < self.full_structures_max:
            self.train_model(all_data + self.transfer_data)
        elif self._ready_state and isinstance(all_data, list) and len(all_data) > self.full_structures_max:
            selected = np.zeros(self.full_structures_max).astype(int)
            sort_indexs = np.argsort([s.get_potential_energy() for s in all_data])
            selected[0:self.full_structures_best] = sort_indexs[0:self.full_structures_best] # Take num_best of the lowest energy structures.
            # Randomly take some of the remaining structures. 
            full_structures_random = self.full_structures_max - self.full_structures_best
            selected[self.full_structures_best:] = np.random.choice(sort_indexs[self.full_structures_best::], size=full_structures_random, replace=False) 
            selected_data = [all_data[i] for i in selected]
            print('SPARSIFICATION0',len(all_data),self.full_structures_best,full_structures_random,len(selected_data))
            self.train_model(selected_data + self.transfer_data)
        else:
            self.train_model(all_data)
        
        
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

    def get_model_parameters(self):
        parameters = {}
        parameters['alpha'] = self.alpha
        parameters['X_train_'] = self.X_train_
        parameters['single_atom_energies'] = self.single_atom_energies
        parameters['bias_per_atom'] = self.bias_per_atom
        return parameters
    
    def set_model_parameters(self, parameters):
        self.alpha = parameters['alpha']
        self.X_train_ = parameters['X_train_']
        self.single_atom_energies = parameters['single_atom_energies']
        self.bias_per_atom = parameters['bias_per_atom']
        self.set_ready_state(True)
