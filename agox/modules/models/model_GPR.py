import numpy as np
import os
import glob
from time import sleep
from ase.io import read,write
from ase.calculators.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes

from timeit import default_timer as dt

from agox.modules.models.model_ABC import ModelBaseClass

from ase import Atom

from agox.modules.models.gaussian_process.kernels import clone

class ModelGPR(ModelBaseClass):
    """
    Model calculator that trains on structures form ASLA memory.
    """
    implemented_properties = ['energy', 'forces', 'uncertainty', 'force_uncertainty']

    name = 'ModelGPR'
    
    def __init__(self, model=None, max_training_data=None, episode_start_training=7, update_interval=1, max_energy=1000, max_adapt_iters=0, n_adapt=25, 
                    force_kappa=0, extrapolate=False, optimize_loglikelyhood=True, verbose=True, use_saved_features=False, sparsifier=None,**kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.episode_start_training = episode_start_training
        self.update_interval = update_interval
        self.max_energy = max_energy
        self.max_training_data = max_training_data
        self.optimize_loglikelyhood = optimize_loglikelyhood
        self.max_adapt_iters = max_adapt_iters
        self.n_adapt = n_adapt
        self.use_saved_features = use_saved_features

        self.force_kappa = force_kappa
        self.extrapolate = extrapolate
        self.sparsifier = sparsifier
        self.verbose = verbose

        # Important for use with DeltaModel
        self.part_of_delta_model = False

        self.model.verbose = self.verbose # This may cause somewhat unexpected print out behaviour, but deal with it.
               
    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        E, E_err = self.predict_energy(self.atoms, return_uncertainty=True)
        self.results['energy'] = E
        self.results['uncertainty'] = E_err
        
        if 'forces' in properties:
            forces,f_err = self.predict_forces(self.atoms, return_uncertainty=True)
            self.results['forces'] = forces
            self.results['force_uncertainty'] = f_err

    def predict_energy(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        if return_uncertainty:
            E, E_err,_ = self.model.predict_energy(atoms, fnew=feature, return_error=True)
            return E, E_err
        else:
            E = self.model.predict_energy(atoms, fnew=feature, return_error=False)
            return E

    def predict_forces(self, atoms, feature=None, return_uncertainty=False, **kwargs):
        if return_uncertainty:
            F, F_err = self.model.predict_force(atoms, fnew=feature, return_error=True)
            return F.reshape(len(atoms), 3), F_err.reshape(len(atoms), 3)
        else:
            F = self.model.predict_force(atoms, fnew=feature).reshape(len(atoms), 3)
            return F
        
    def batch_predict(self, data, over_write=False):        
        E = np.array([self.model.predict_energy(atoms, return_error=False) for atoms in data])
        if over_write:
            self.batch_assign(data, E)

        return E

    ####################################################################################################################
    # Training:
    ####################################################################################################################

    def pretrain(self, structures):
        energies = [atoms.get_potential_energy() for atoms in structures]
        self.model.train(structures)
        self.pretrain_structures = structures

        print('Model pretrained on {} structures'.format(len(structures)))
        print('Lowest energy: {} eV'.format(np.min(energies)))
        print('Median energy {} eV'.format(np.mean(energies)))
        print('Max energy {} eV'.format(np.max(energies)))
        self.set_ready_state(True)

    def train_model(self, all_data, energies):
        t0 = dt()
        
        # Train on the best structures:
        args = np.argsort(energies)
        E_best = energies[args[0]]        
        allowed_idx = [arg for arg in args if energies[arg] - E_best < self.max_energy]

        if self.max_training_data is not None:
            allowed_idx = allowed_idx[0:self.max_training_data]
            
        remaining = [arg for arg in args if arg not in allowed_idx]

        training_data = [all_data[i] for i in allowed_idx]
        training_energies = [energies[i] for i in allowed_idx]

        if self.verbose:
            print(f'Min GPR training energy: {np.min(training_energies)}')
            print(f'Max GPR training energy: {np.max(training_energies)}')
        # Wipe any stored atoms object, since now the model will change and hence give another result
        self.atoms = None

        # Use saved features: 
        if self.use_saved_features:
            features = []
            deltas = []
            for candidate in training_data:
                F = candidate.get_meta_information('GPR_feature')
                d = candidate.get_meta_information('GPR_delta')
                if F is None:
                    F = self.model.featureCalculator.get_feature(candidate)
                    candidate.add_meta_information('GPR_feature', F)
                    d = self.model.delta_function.energy(candidate)
                    candidate.add_meta_information('GPR_delta', d)
                features.append(F)
                deltas.append(d)
            features = np.array(features)
            deltas = np.array(deltas)
            print(F.shape, deltas.shape, np.array(training_energies).shape)
            self.model.train(features=features, data_values=training_energies, delta_values=deltas, add_new_data=False, optimize=self.optimize_loglikelyhood)                
        else:
            self.model.train(training_data, data_values=training_energies, add_new_data=False, optimize=self.optimize_loglikelyhood)

        self.set_ready_state(True) # Should only use this. 
        if self.max_adapt_iters > 0 and len(remaining) > 0:
            print('Running adaptive training on remaining {}'.format(len(remaining)))
            self.adaptive_training(all_data, energies, remaining)

        t1 = dt()
        if self.verbose:
            print('GPR Training time: {}'.format(t1-t0))

    def adaptive_training(self, all_data, energies, remaining):
        run_adapt = True
        t0 = dt()
        adapt_iters = 0
        E_best = np.min(energies)
        while run_adapt and self.n_adapt > 0:

            if len(remaining) > 0:

                # Cases:
                # 1: Predict outside, True outside
                # 2: Predict inside,  True outside --> Worst case
                # 3: Predict outside, True inside  --> Model predicts a too high energy. 
                # 4: Predict inside,  True inside

                # 1: cond2 = False, cond1 = false
                # 2: cond2 = True,  cond1 = False
                # 3: cond2 = False, cond1 = True
                # 4: cond2 = True,  cond2 = True

                # Get prediction energies:
                true_energies = np.array([energies[idx] for idx in remaining])
                predicted_energies = np.array([self.model.predict_energy(all_data[i], return_error=False) for i in remaining])
                error = true_energies - predicted_energies
             
                # Conditions:
                cond1 = (true_energies - E_best < self.max_energy) # True if true_energy within max_energy of best, False otherwise
                cond2 = (predicted_energies - E_best < self.max_energy) # True if predicted energy within max_energy of best, False otherwise
                cond3 = error > self.max_energy / 4 # True if difference larger than max_energy / 4
                total_cond = (cond1 + cond2) * cond3

                non_zero_prob_idx = np.argwhere(total_cond == True)

                if len(non_zero_prob_idx) == 0:
                    break

                # Uniform probability for all non-discarded structures:
                probs = np.zeros(len(remaining))
                probs[non_zero_prob_idx] = 1
                probs = probs / np.sum(probs)
                
                # Pick the new training data:
                strucs_left = len(np.argwhere(probs != 0))
                if self.n_adapt <= strucs_left:
                    choices = np.random.choice(remaining, size=self.n_adapt, p=probs, replace=False)
                else:
                    choices = np.random.choice(remaining, size=strucs_left, p=probs, replace=False)
                    
                # Train on it
                adapt_batch = [all_data[c] for c in choices]
                data_values = [energies[c] for c in choices]
                self.model.train(adapt_batch, data_values=data_values, add_new_data=True)

                # Remove choices from remaining:
                remaining = [i for i in remaining if i not in choices]

                # When to stop:
                if (total_cond == False).all():
                    break
                elif adapt_iters >= self.max_adapt_iters:
                    break
                
                adapt_iters += 1 

            else:
                break

        t1 = dt()
        print('Model trained on {} structures'.format(len(all_data)-len(remaining)))
        print('Model training took {:4.2f} s'.format(t1-t0))
                    
    def training_observer_func(self, database):
        episode = self.get_episode_counter()
        if episode < self.episode_start_training:
            return
        if (episode % self.update_interval != 0) * (episode != self.episode_start_training):
            return 

        # Find complete builds:
        all_data = database.get_all_candidates()
        if self.sparsifier != None:
            _, all_data = self.sparsifier(all_data)
        energies = np.array([x.get_potential_energy() for x in all_data])
        self.train_model(all_data, energies)

    ####################################################################################################################
    # For MPI Relax. 
    ####################################################################################################################

    def set_verbosity(self, verbose):
        self.verbose = verbose
        self.model.verbose = verbose

    def get_model_parameters(self):
        parameters = {}
        parameters['feature_mat'] = self.model.featureMat
        parameters['alpha'] = self.model.alpha
        parameters['bias'] = self.model.bias
        parameters['kernel_hyperparameters'] = self.model.kernel_.get_params()
        parameters['K_inv'] = self.model.K_inv
        return parameters
    
    def set_model_parameters(self, parameters):

        self.model.featureMat = parameters['feature_mat']
        self.model.alpha = parameters['alpha']
        self.model.bias = parameters['bias']
        self.model.K_inv = parameters['K_inv']

        try:
            self.model.kernel_ = clone(self.model.kernel_)
        except:
            self.model.kernel_ = clone(self.model.kernel)

        self.model.kernel_.set_params(**parameters['kernel_hyperparameters'])
        self.set_ready_state(True)

    ####################################################################################################################
    # Misc. 
    ####################################################################################################################

    def assign_from_main(self, main):
        super().assign_from_main(main)
