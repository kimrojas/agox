from ase import atoms
import numpy as np

from .model_ABC import ModelBaseClass
from ase.calculators.calculator import all_changes
from ase.calculators.test import numeric_force

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
import scipy.optimize

class GlobalKernel:

    def __init__(self, local_kernel, descriptor=None):
        self.local_kernel = local_kernel

    def __call__(self, features_N, features_M=None):
        if features_M is None:
            return self.get_global_kernel(features_N, features_N, symmetric=True)
        else:
            return self.get_global_kernel(features_N, features_M)

    def get_global_kernel(self, features_N, features_M, symmetric=False):
        """
        The global kernel is the covariance between the atoms-objects

        kernel size: |data_N| x |data_M| 

        |x| = len(x)
        """

        N = features_N.shape[0]
        M = features_M.shape[0]
        K = np.zeros((N, M))

        if not symmetric:
            for i in range(N):
                for j in range(M):
                    K[i, j] = self.get_global_covariance(features_N[i], features_M[j])
        
        elif symmetric:
            for i in range(N):
                for j in range(i, N):
                    K[i, j] = self.get_global_covariance(features_N[i], features_M[j])
            K = K + K.T - np.diag(np.diag(K))
        return K

    def get_global_covariance(self, feature_i, feature_j):
        """
        Takes local feature vectors and calculates the global covariance

        Local feature vector:
        size: N_atoms x N_features
        """
        C = np.sum(self.local_kernel(feature_i, feature_j))
        
        return C

class LocalGPRModel(ModelBaseClass):

    name = 'localGPR'
    implemented_properties = ['energy', 'forces'] # 'uncertainty']

    """
    Implements a local Gaussian Process Model

    The key feature of such a model is that the covariance-matrix is a sum of local covariances (or kernel matrices.)
    """
    def __init__(self, kernel, noise=1e-10, descriptor=None, max_training_data=1E10, num_best=0, **kwargs):
        super().__init__(**kwargs)
        self.noise = noise
        self.kernel_ = kernel
        self.descriptor = descriptor

        # Training Arguments:
        self.max_training_data = max_training_data
        self.num_best = num_best
        self.num_random = max_training_data - num_best

    ################################## ##################################################################################
    # Prediction
    ####################################################################################################################

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results['energy'] = self.predict_energy(atoms_list=[atoms])[0]

        if 'forces' in properties:
            self.results['forces'] = self.predict_forces_V2(atoms_list=[atoms])
    
    def predict_energy(self, X=None, atoms_list=None):
        if X is not None:
            pass
        elif X is None and atoms_list is not None:
            X = np.array([self.descriptor(atoms) for atoms in atoms_list])
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        K_trans = self.kernel_(X, self.X_train_)
        y_mean = K_trans.dot(self.alpha)  # Line 4 (y_mean 

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
                        k = self.kernel_.local_kernel(x[l].reshape(1,-1), self.X_train_[n][j].reshape(1,-1))
                        e_local[l] += self.alpha[n] * np.sum(k)
            y_mean.append(e_local)
            
            if return_unc:
                e_local_unc = np.zeros(atoms_shape)
                for l in range(atoms_shape):
                    k0 = self.kernel_.local_kernel(x[l].reshape(1,-1),x[l].reshape(1,-1))
                    k = np.zeros(self.X_train_.shape[0])
                    for j in range(self.X_train_.shape[0]):
                        for i in range(self.X_train_.shape[1]):
                            k[j] += self.kernel_.local_kernel(x[l].reshape(1,-1),self.X_train_[j][i].reshape(1,-1))
                    vk = np.dot(self._K_inv,k.reshape(-1))

                    e_local_unc[l] = np.sqrt(k0 - np.dot(k,vk))
                y_mean_unc.append(e_local_unc)
                
        if return_unc:
            return y_mean, y_mean_unc
        else:
            return y_mean

    def predict_uncertainty(self, X=None, atoms_list=None):
        if X is not None:
            pass
        elif X is None and atoms_list is not None:
            X = np.array([self.descriptor(atoms) for atoms in atoms_list])
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        uncertainty = []
        for x in X:
            x_expanded = np.expand_dims(x,0)
            k0 = self.kernel_(x_expanded)
            k = self.kernel_(x_expanded, self.X_train_)
            vk = np.dot(self._K_inv,k.reshape(-1))
            uncertainty.append(np.sqrt(np.sum(k0 - np.dot(k,vk))))

        return uncertainty
    
    def predict_forces(self, X=None, atoms_list=None):
        """
        Does not work with several structures yet.

        Is numerical. Dont hate
        """
        # if X is not None:
        #     pass
        # # elif X is None and atoms_list is not None:
        # #     X = np.array([self.descriptor(atoms) for atoms in atoms_list])
        # else:
        #     print('Need to specifiy either feature matrix X or atoms_list')
        #     print('I will break in a moment')

        d = 0.001
        return np.array([[numeric_force(atoms_list[0], a, i, d) for i in range(3)] for a in range(len(atoms_list[0]))])

    def predict_forces_V2(self, X=None, atoms_list=None):
        """
        Faster than the other method, but still numerical so not lightning speed.
        """
        atoms = atoms_list[0].copy()

        d = 0.001

        all_atoms = []
        for a in range(len(atoms)):
            for i in range(3):
                patoms = atoms.copy()
                patoms.positions[a, i] += d
                matoms = atoms.copy()
                matoms.positions[a, i] -= d

                all_atoms.extend([patoms, matoms])

        X = np.array([self.descriptor(atoms) for atoms in all_atoms])

        energies = self.predict_energy(X=X)

        penergies = energies[0::2]
        menergies = energies[1::2]

        forces = ((menergies - penergies) / (2 * d)).reshape(len(atoms), 3)
        return forces

    def batch_predict(self, data, over_write=False):
        energies = self.predict_energy(atoms_list=data)

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

        if len(atoms_list) > self.max_training_data:
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
                y = [atoms.get_potential_energy() for atoms in atoms_list]
        else:
            print('Need to specifiy either feature matrix X or atoms_list')
            print('I will break in a moment')

        self.X_train_ = np.copy(X) #if self.copy_X_train else X
        self.y_train_ = np.copy(y) #if self.copy_X_train else y

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.noise
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            # self._K_inv = None
            self._K_inv = cho_solve((self.L_, True), np.eye(K.shape[0]))
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha = self._K_inv @ self.y_train_ # cho_solve((self.L_, True), self.y_train_)  # Line 3
        return self.alpha

    def train_model(self, training_data, energies):
        self.set_ready_state(True)
        self.train_GPR(atoms_list=training_data, y=energies)

    ####################################################################################################################
    # Assignments:
    ####################################################################################################################

    def assign_from_main(self, main):
        super().assign_from_main(main)
        if not self.part_of_delta_model:
            main.database.attach_observer(self.name, self.observer_func)

    def observer_func(self, database):
        
        candidates = self.get_all_candidates()
        y = [a.get_potential_energy() for a in candidates]

        self.train_model(candidates, y=y)

    ####################################################################################################################
    # Misc.
    ####################################################################################################################

    def save_model(self, directory='', prefix=''):
        np.save(directory+prefix+'alpha.npy', self.alpha)
        np.save(directory+prefix+'training_features.npy', self.X_train_)
    
    def load_model(self, directory='', prefix=''):
        """
        Loads saved model weights and features, but does not check that you are using the 
        same kernel or hyperparameters! 
        """
        self.alpha = np.load(directory+prefix+'alpha.npy')
        self.X_train_ = np.load(directory+prefix+'training_features.npy')

    
