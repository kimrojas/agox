import numpy as np
import os
import glob
from time import sleep
from ase.io import read,write
from ase.calculators.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes

from timeit import default_timer as dt

from ase import Atom

class ModelCalculator(Calculator):
    """
    Model calculator that trains on structures form ASLA memory.
    """

    implemented_properties = ['energy', 'forces', 'uncertainty']
    
    def __init__(self, model=None, kappa=0, episode_start_training=7, train_N_best=500, train_filter=False, dE=5, update_interval=1, max_energy=20, 
                 test_set=None, episode_stop_training=1E10, max_adapt_iters=5, n_adapt=25, **kwargs):
        self.results = {}
        self.model = model
        self.kappa = kappa
        self.episode_start_training = episode_start_training
        self.train_N_best = train_N_best
        self.train_filter = train_filter
        self.dE = dE
        self.update_interval = update_interval
        self.max_energy = max_energy
        self.test_set = test_set

        self.ever_trained = False
        self.pretrain_structures = None

        self.episode_stop_training = episode_stop_training

        self.update_func = self.update_V1

        # Settings for update_V2:
        self.max_adapt_iters = max_adapt_iters
        self.n_adapt = n_adapt

        # Settings for update_V3:
        self.best_energy_when_trained = 0
        
        super().__init__(**kwargs)
        
    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes,
                  one_time_kappa=None):

        if one_time_kappa is not None:
            kappa = one_time_kappa
        else:
            kappa = self.kappa

        #print('kappa',kappa)

        Calculator.calculate(self, atoms, properties, system_changes)
        self.atoms = atoms.copy()
        self.atoms.pbc = [1,1,1]

        E, E_err,_ = self.model.predict_energy(self.atoms, return_error=True)
        #print('E E_err',E, E_err)
        self.results['energy'] = E - kappa*E_err
        self.results['uncertainty'] = E_err
        
        if 'forces' in properties:
            forces,f_err = self.model.predict_force(self.atoms,return_error = True)
            F = (forces+self.kappa*f_err).reshape(len(self.atoms),3)
            for c in self.atoms.constraints:
                c.adjust_forces(self.atoms,F)
            self.results['forces'] = F

    def _get_filtered_training_data(self,X):
        # X must be sorted according to energy
        E = X[0].get_potential_energy()
        dE = self.dE
        i = 0
        js = [0]
        while i < len(X):
            if X[i].get_potential_energy() > E + dE:
                js.append(i)
                E += dE
                dE *= 2
            i += 1

        js.append(len(X))

        probs = np.zeros(len(X))
        p0 = 0.2
        for i, (imin, imax) in enumerate(zip(js[:-1],js[1:])):
            n = imax - imin
            print(i,imin,imax,p0,p0**i,p0**i/n)
            probs[imin:imax] = p0**i/n #* np.ones(n)

        Nprobs = np.sum(probs)
        assert Nprobs > 0, 'not possible'
        probs /= Nprobs

        sample_size = int((js[1] - js[0]) / (1 - p0)) # so structure with Ebest <= E <= Ebest + dE gets to take up a major share of structures
        sample_size = max(sample_size,1)
        if self.train_N_best == 'half':
            sample_size = sample_size//2
        elif self.train_N_best == 'three quarters':
            sample_size = sample_size//4 * 3
        else:
            sample_size = min(sample_size,self.train_N_best) # 1 <= sample_size <= self.train_N_best
        sample_size = min(sample_size,len(X))

        sample_indices = np.random.choice(len(X),size=sample_size,p=probs,replace=False)
        sample_indices.sort()

        return [X[i] for i in sample_indices]

    def pretrain(self, structures):
        energies = [atoms.get_potential_energy() for atoms in structures]
        self.model.train(structures)
        self.pretrain_structures = structures

        print('Model pretrained on {} structures'.format(len(structures)))
        print('Lowest energy: {} eV'.format(np.min(energies)))
        print('Median energy {} eV'.format(np.mean(energies)))
        print('Max energy {} eV'.format(np.max(energies)))
        self.ever_trained = True

    def update(self):
        self.update_func()

    def update_V1(self):
        episode_no = self.get_episode_counter()
        if episode_no < self.episode_start_training or episode_no % self.update_interval != 0:
            return
        
        if episode_no > self.episode_stop_training:
            print('No longer training model')
            return 

        #X = [g for g in self.memory.get_all_finished_structures()]
        X = [g for g in self.new_memory.get_all_candidates()]

        if len(X) == 0:
            print('Not enough structure yet for training model')
            return
        
        #for i,x in enumerate(X):
        #    print('X0',i,len(x),x.get_potential_energy())

        print('update1_',self.train_N_best,len(X))

        # find complete builds among all structures
        max_len = np.max([len(x) for x in X])
        X = [x for x in X if len(x)==max_len]
        #for i,x in enumerate(X):
        #    print('X1',i,len(x),x.get_potential_energy())

        print('update1a',self.train_N_best,len(X))

        # sort and pick those under a cutoff measured from the best
        args = np.argsort([x.get_potential_energy() for x in X])
        E_best = X[args[0]].get_potential_energy()
        print('args0',args[0])
        print('E_best',E_best)
        X = [X[arg] for arg in args if X[arg].get_potential_energy() - E_best < self.max_energy]

        print('update1b',self.train_N_best,len(X))
        for x in X[:5]:
            print('update2a',x.get_potential_energy())
        print('update2b ..')
        for x in X[-5:]:
            print('update2c',x.get_potential_energy())

        if self.train_N_best is not None:
            if self.train_filter:
                X = self._get_filtered_training_data(X)
            else:
                if self.train_N_best == 'half':
                    X = X[:len(X)//2]
                elif self.train_N_best == 'three quarters':
                    X = X[:len(X)//4 * 3]
                else:
                    X = X[:self.train_N_best]
            print('update1c',self.train_N_best,len(X))
            print('update2d ..')
            for x in X[-5:]:
                print('update2e',x.get_potential_energy())

        self.model.train(X, add_new_data=False)

        try:
            k1 = self.model.kernel_.get_params()['k1__k1']
            l1 = self.model.kernel_.get_params()['k1__k2__k1__k2__length_scale']
            l2 = self.model.kernel_.get_params()['k1__k2__k2__k2__length_scale']
            print('L1: {:<20} L2: {:<20}K1: {:<20}'.format(l1,l2,str(k1)))
        except:
            print('Cannot print kernel HPs. Continuing')

        self.ever_trained = True

        self.test_model()

    def update_V2(self):
        
        episode_no = self.get_episode_counter()
        if episode_no < self.episode_start_training or episode_no % self.update_interval != 0:
            return

        # Find complete builds:
        X = [g for g in self.memory.get_all_data()]
        max_len = np.max([len(x) for x in X])
        all_data = [x for x in X if len(x)==max_len]
        energies = np.array([x.get_potential_energy() for x in all_data])

        print(np.min(energies))
        print(np.max(energies))

        # Train on the best structures:
        args = np.argsort(energies)
        E_best = energies[args[0]]        
        allowed_idx = [arg for arg in args if energies[arg] - E_best < self.max_energy]

        if self.train_N_best is not None:
            allowed_idx = allowed_idx[0:self.train_N_best]
            
        remaining = [arg for arg in args if arg not in allowed_idx]

        training_data = [all_data[i] for i in allowed_idx]

        # Train on these:
        self.model.train(training_data, add_new_data=False)
        self.ever_trained = True
        run_adapt = True

        t0 = dt()
        adapt_iters = 0
        while run_adapt: 

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
                self.model.train([all_data[c] for c in choices], add_new_data=True)

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

    def update_V3(self):
        episode_no = self.get_episode_counter()
        if episode_no < self.episode_start_training or episode_no % self.update_interval != 0:
            return

        best_energy = self.memory.get_best_energy()

        retrain_threshold = -10
        # Completely retrain the model: 
        if best_energy - self.best_energy_when_trained < retrain_threshold:
            t0 = dt()

            # Find complete builds:
            X = [g for g in self.memory.get_all_data()]
            max_len = np.max([len(x) for x in X])
            all_data = [x for x in X if len(x)==max_len]
            energies = np.array([x.get_potential_energy() for x in all_data])

            # Train on the best structures:
            all_indexs = np.argsort(energies)
            allowed_idx = [idx for idx in  all_indexs if energies[idx] - best_energy < self.max_energy]

            if self.train_N_best is not None:
                allowed_idx = allowed_idx[0:self.train_N_best]
                
            remaining = [idx for idx in all_indexs if idx not in allowed_idx]

            training_data = [all_data[i] for i in allowed_idx]

            # Train on these:
            self.model.train(training_data, add_new_data=False)
            self.ever_trained = True

            amount_of_data = len(training_data)

            for index in remaining:
                add_to_model = True
                grid = all_data[index]
                energy = energies[index]

                epred = self.model.predict_energy(grid, return_error=False)
                print(epred, energy)

                pred_zone = epred-best_energy
                real_zone = energy-best_energy

                print(pred_zone, real_zone, epred)

                # Both outside:
                if pred_zone > self.max_energy and real_zone > self.max_energy: 
                    add_to_model = False
                elif np.abs(energy-epred) < 2:
                    add_to_model = False
                
                if add_to_model:
                    self.model.train([grid], add_new_data=True)
                
                    amount_of_data += 1
            print('Model trained on {} structures.'.format(amount_of_data))
            print('Training took: {} s'.format(dt()-t0))
        else:

            energies, grids = self.memory.get_most_recent_data(self.update_interval)

            for grid, energy in zip(grids, energies):
                add_to_model = True
                epred = self.model.predict_energy(grid, return_error=False)

                pred_zone = epred-best_energy
                real_zone = energy-best_energy

                # Both outside:
                if pred_zone > self.max_energy & real_zone > self.max_energy: 
                    add_to_model = False
                # Large error
                elif np.abs(energy-pred) < 2:
                    add_to_model = False
                
                if add_to_model:
                    self.model.train([grid], add_new_data=True)
                
    def test_model(self):
        if self.test_set is None:
            return

        true_energies = np.array([atoms.get_potential_energy() for atoms in self.test_set])
        predicted_energies = np.zeros(len(true_energies))
        for i, a in enumerate(self.test_set):
            atoms = a.copy()
            atoms.set_calculator(self)
            predicted_energies[i] = atoms.get_potential_energy()
        error_w_sign = predicted_energies - true_energies
        abs_error = np.abs(error_w_sign)

        print('Model target  [',','.join(['{:8.3f}'.format(x) for x in true_energies]),']')
        print('Model predict [',','.join(['{:8.3f}'.format(x) for x in predicted_energies]),']')
        print('Model error   [',','.join(['{:8.3f}'.format(x) for x in error_w_sign]),']')
    
        
    def assign_from_ASLA(self, asla, main):
        self.memory = asla.memory
        self.new_memory = asla.new_memory
        self.get_episode_counter = asla.get_episode_counter
        self._prepare_for_completing_grids(asla.builder, asla.grid)

    def _put_missing_atoms_at_infinity(self, grid):
        dx = 100
        x = dx
        for t in self.atom_types:
            while sum(grid.numbers==t) < self.atoms_this_type_in_complete_structure[t]:
                grid.extend(Atom(t,[x,0,0]))
                x += dx
        return grid

    def _prepare_for_completing_grids(self, builder, template):
        self.atom_types = builder.atom_types
        self.atoms_this_type_in_complete_structure = {}
        N_total = 0
        for t in self.atom_types:
            self.atoms_this_type_in_complete_structure[t] = \
                                sum(template.numbers==t) + sum([1 for n in builder.numbers if n==t])
            N_total += self.atoms_this_type_in_complete_structure[t]
        self.N_total = N_total

# Cases:
# 1: Predict outside, True outside
# 2: Predict inside,  True outside --> Worst case
# 3: Predict outside, True inside  --> Model predicts a too high energy. 
# 4: Predict inside,  True inside


# Case 1: Discard because this structure is uninteresting. 
# Case 2: Train 
# Case 3: Train
# Case 4: Only train of large error. 
