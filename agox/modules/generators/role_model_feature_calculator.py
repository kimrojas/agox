from agox.modules.generators.ABC_generator import GeneratorBaseClass
import numpy as np
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
import random
from scipy.optimize import minimize
from ase import Atoms

class RolemodelGeneratorFC(GeneratorBaseClass):

    name = 'RolemodelGeneratorFC'

    def __init__(self, 
                possible_attractors = [3, 6], 
                move_all=1, 
                lambs = [1], 
                rc = 11.9, 
                number_of_mutations=1, 
                rm=[], 
                rolemodels_from_template = 1,
                feature_calculator = None,
                **kwargs):

        super().__init__(**kwargs)
        self.possible_attractors = possible_attractors
        self.move_all = move_all
        self.lambs = lambs
        self.rc = rc
        self.number_of_mutations = number_of_mutations
        self.rm = rm
        self.rolemodels_from_template = rolemodels_from_template
        self.feature_calculator = feature_calculator

    def get_candidates(self, sampler, environment):
        candidate = sampler.get_random_member()

        # Happens if no candidates are part of the sample yet. 
        if candidate is None:
            return [None]
        
        template = candidate.get_template()
        n_template = len(template)

        # Function that minimizes CE
        def objective_func(pos):
            candidate.positions[mover_indices,:] = np.reshape(pos,[len(mover_indices),3])
            features = self.calculate_features(candidate)
            ce = self.calculate_ce(features[mover_indices], role_models)
#            print('ce',ce)
            grad = self.calculate_numerical_ce_gradients(candidate, mover_indices, role_models,0.00001)
#            grad = self.ce_gradient(candidate, features, mover_indices, role_models)
            return ce, grad.flatten()

        # Calculate features
        features = self.calculate_features(candidate)

        # Set role models
        if len(self.rm) != 0:
            role_models = self.rm
        else:
            if len(self.possible_attractors) == 0:
                n_attractors = self.possible_attractors[0]
            else:
                n_attractors = np.random.randint(self.possible_attractors[0], self.possible_attractors[1] + 1)
            
            if self.rolemodels_from_template:
                indices = random.sample(range(len(features)), k=n_attractors)
            else:
                indices = random.sample(range(n_template, len(features)), k=n_attractors)
            role_models = features[indices]

        # Determine which atoms to move
        if self.move_all == 1:
            mover_indices = range(n_template, len(candidate))
        else:
            n_movers = np.random.randint(1, len(candidate) - n_template)
            ce, local_ce = self.calculate_ce(features, role_models, local = 1)
            largest_local_ce = local_ce[n_template:] # sort out template
            largest_local_ce_indices = np.argsort(largest_local_ce)[::-1]
            mover_indices = largest_local_ce_indices[:n_movers] + n_template

        # Minimize CE
        pos = candidate.positions[mover_indices].flatten()
        pos = minimize(objective_func, pos, method = 'BFGS', jac = True, options={'maxiter':75}).x
        suggested_positions = np.reshape(pos,[len(mover_indices), 3])

        for index in range(len(mover_indices)):
            i = mover_indices[index]
            for _ in range(100):
                if _ == 0:
                    radius = 0
                else:
                    radius = 0.5 * np.random.rand()**(1/self.dimensionality)
                displacement = self.get_displacement_vector(radius)
                suggested_position = suggested_positions[index] + displacement

                # Check confinement limits:
                if not self.check_confinement(suggested_position):
                    continue

                # Check that suggested_position is not too close/far to/from other atoms
                # Skips the atom it self. 
                if self.check_new_position(candidate, suggested_position, candidate[i].number, skipped_indices=[i]):
                    candidate[i].position = suggested_position 
                    break

        candidate = self.convert_to_candidate_object(candidate, template)
        candidate.add_meta_information('description', self.name)
        return [candidate]

    def gc(self, r):
        filter = r < self.rc
        values = np.zeros(len(r))
        values[filter] = 0.5 * np.cos(np.pi * r[filter] / self.rc) + 0.5
        return values

    def calculate_features(self, struc):
        features = self.feature_calculator.get_feature(struc)
        return features

#    def calculate_features(self, struc):
#        a_numbers = struc.get_atomic_numbers()
#        n_types = sorted(list(set(a_numbers)))
#
#        nl = NeighborList([self.rc / 2] * len(struc), self_interaction = 0, bothways = 1)
#        nl.update(struc)
#
#        features = []
#        
#        for i in range(len(struc)):
#            feature = []
#            ind, uc = nl.get_neighbors(i)
#            pos = struc.positions[ind] + np.dot(uc, struc.get_cell())
#            dists = np.linalg.norm(struc.positions[i] - pos, axis=1)
#            for j in n_types:
#                for lamb in self.lambs:
#                    filter = struc.numbers[ind] == j
#                    rho = np.sum(np.exp(- dists[filter] / lamb) * self.gc(dists[filter])) / lamb
#                    feature.append(rho)
#            feature.append(struc.numbers[i])
#            features.append(feature)
#        return np.array(features)

    def calculate_ce(self, features, role_models, local = 0):
        dists = cdist(features, role_models)
        indices = np.argmin(dists, axis=1)
        local_ce = np.array([dists[i][indices[i]] for i in range(len(indices))])
        ce = sum(local_ce)
        if local:
            return ce, local_ce
        else:
            return ce

    def ce_gradient(self, struc, features, mover_indices, role_models):
        grad = np.zeros([len(struc), 3])

        types = sorted(list(set(struc.get_atomic_numbers())))
        mover_rolemodel_index = np.argmin(cdist(features[mover_indices], role_models), axis = 1)
        feature_dvec = features[mover_indices] - role_models[mover_rolemodel_index]
        feature_dists = np.linalg.norm(feature_dvec, axis = 1)

        nl = NeighborList([self.rc / 2] * len(struc), self_interaction = 0, bothways = 1)
        nl.update(struc)

        for a, i in enumerate(mover_indices):
            if feature_dists[a] != 0:
                type_index = np.argmax(types == struc.numbers[i])
                index, uc = nl.get_neighbors(i)

                pos = struc.positions[index] + np.dot(uc, struc.get_cell())
                dvecs = struc.positions[i] - pos
                dists = np.linalg.norm(dvecs, axis = 1)

                for h, lamb in enumerate(self.lambs):
                    scalar_vec = np.exp(-dists / lamb) / (2 * dists * feature_dists[a]) / lamb
                    scalar_vec *= - (1/lamb + 1/lamb * np.cos(np.pi * dists / self.rc) + np.pi / self.rc * np.sin(np.pi * dists / self.rc))
                    scalar_vec = np.expand_dims(scalar_vec, axis = 1) * dvecs 

                    for b, type in enumerate(types):
                        b *= len(self.lambs)
                        filter = struc.numbers[index] == type
                        scalar_vec[filter] *= feature_dvec[a][b+h]

                    for k in mover_indices:
                        args = np.where(index == k)
                        sv = scalar_vec[args]
                        sv = np.sum(sv, axis = 0)
                        grad[k] -= sv

                    scalar_vec = np.sum(scalar_vec, axis = 0)
                    grad[i] += scalar_vec

        return grad[mover_indices]

    # Numerical gradient
    def calculate_numerical_ce_gradients(self, candidate, mover_indices, role_models,  d=0.0001):
        return np.array([[self.numeric_ce_gradient(candidate, role_models, mover_indices, a, i, d)
                          for i in range(3)] for a in mover_indices])

    def numeric_ce_gradient(self, candidate, role_models, mover_indices, a, i, d=0.001):
        mover_indices = list(mover_indices)
        p0 = candidate.get_positions()
        p = p0.copy()
        p[a, i] += d
        candidate.set_positions(p, apply_constraint=False)
        features = self.calculate_features(candidate)
        ce_plus = self.calculate_ce(features[mover_indices], role_models)

        p[a, i] -= 2 * d
        candidate.set_positions(p, apply_constraint=False)
        features = self.calculate_features(candidate)
        ce_minus = self.calculate_ce(features[mover_indices], role_models)

        candidate.set_positions(p0, apply_constraint=False)

        return (ce_plus - ce_minus) / (2 * d)