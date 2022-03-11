from agox.modules.models.model_ABC import ModelBaseClass
from ase.calculators.calculator import Calculator, all_changes
from ase.data import covalent_radii

from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList

import numpy as np

from scipy.optimize import fmin, minimize


class ModelLennardJones(ModelBaseClass):

    name = 'LJTheFabolousModel'

    implemented_properties = ['energy', 'forces']

    """
    Implements trainable Lennard-Jones potential. 

    E(r_ab) = 4 * eps_ab * [(sigma_ab / r_ab)^12 - (sigma_ab/r_ab)^6]

    Where r_ab is the distance between atoms a and b. There are two parameters per. interaction-type: 

    eps_ab: Strength/depth of the potential. 

    sigma_ab: Location of E = 0 (So E(sigma_ab) = 0) on the 'compressed' side of the potential. 

    For n different types there will be n*(n+1)/2 of each parameter (triangular numbers), corresponding to 
    'half' + the diagonal of a matrix of size n*n. 
    """

    def __init__(self, rc=10, **kwargs):
        self.nl = None
        self.rc = rc
        super().__init__(**kwargs)

    def assign_from_main(self, main):
        missing_types = np.unique(main.environment.get_numbers())
        template_types = main.environment.get_template().get_atomic_numbers()
        self.all_types = np.sort(np.unique(np.append(missing_types, template_types)))
        self.n_types = len(self.all_types)
        self.num_interactions = self.n_types * (self.n_types + 1) // 2

        self.number_to_idx = {self.all_types[i]:i for i in range(self.n_types)}

        #self.epsilon_parameters = np.ones((self.n_types, self.n_types))
        
        epsilon_parameters = np.zeros((self.n_types, self.n_types))
        sigma_parameters = np.zeros((self.n_types, self.n_types))

        self.iidxs = []
        self.jidxs = []
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                sigma_parameters[i, j] = covalent_radii[self.all_types[i]] + covalent_radii[self.all_types[j]]
                epsilon_parameters[i, j] = np.sqrt(self.all_types[i] * self.all_types[j]) # sqrt(Zi*Zj)
                self.iidxs.append(i)
                self.jidxs.append(j)
        self.sigma_parameters = sigma_parameters + sigma_parameters.T - np.diag(np.diag(sigma_parameters))        
        self.epsilon_parameters = epsilon_parameters + epsilon_parameters.T - np.diag(np.diag(epsilon_parameters))

    ####################################################################################################################
    # Prediction Methods:
    ####################################################################################################################    

    def calculate_old_not_stolen_no_force(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        r = cdist(atoms.positions, atoms.positions)
        r[np.diag_indices(len(atoms))] = np.inf

        I, J = self.get_index_matrix(atoms)
        sigma = self.get_sigma_matrix(I, J)
        eps = self.get_epsilon_matrix(I, J)
        
        
        Emat = 4 * eps * ((sigma / r)**12 - (sigma / r)**6)
        E = np.sum(Emat) * 1/2

        self.results['energy'] = E
    
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        energy, forces, energies = self.get_model_properties(atoms=atoms, properties=properties)

        self.results['energy'] = energy
        self.results['energies'] = energies
        self.results['free_energy'] = energy

        if 'forces' in properties:
            self.results['forces'] = forces


    def predict_energy(self, atoms):
        energy, forces, energies = self.get_model_properties(atoms=atoms)
        return energy

    def predict_forces(self, atoms):
        energy, forces, energies = self.get_model_properties(atoms=atoms)
        return forces
        

    def get_model_properties(self, atoms=None, properties=None):
        """
        This is blatantly stolen from ASE. 

        I deleted the stress calculuation, it can be re-added by copying it from the ase LJ calculator..
        """

        if properties is None:
            properties = self.implemented_properties

        natoms = len(atoms)

        rc = self.rc
        # if self.nl is None or 'numbers' in system_changes:
        self.nl = NeighborList([rc / 2] * natoms, self_interaction=False)

        self.nl.update(atoms)

        positions = atoms.positions
        cell = atoms.cell

        # potential value at rc
        #e0 = 4 * epsilon * ((sigma / rc) ** 12 - (sigma / rc) ** 6)
        e0 = 0

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        #stresses = np.zeros((natoms, 3, 3))

        I, J = self.get_index_matrix(atoms)
        sigma_matrix = self.get_sigma_matrix(I, J)
        eps_matrix = self.get_epsilon_matrix(I, J)

        for ii in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(ii)
            cells = np.dot(offsets, cell)

            # pointing *towards* neighbours
            distance_vectors = positions[neighbors] + cells - positions[ii]

            sigma = sigma_matrix[ii, neighbors]
            epsilon = eps_matrix[ii, neighbors]

            r2 = (distance_vectors ** 2).sum(1)
            c6 = (sigma ** 2 / r2) ** 3
            c6[r2 > rc ** 2] = 0.0
            c12 = c6 ** 2

            pairwise_energies = 4 * epsilon * (c12 - c6) - e0 * (c6 != 0.0)
            energies[ii] += 0.5 * pairwise_energies.sum()  # atomic energies


            if 'forces' in properties:
                pairwise_forces = (-24 * epsilon * (2 * c12 - c6) / r2)[
                    :, np.newaxis
                ] * distance_vectors
                forces[ii] += pairwise_forces.sum(axis=0)

            # add j < i contributions
            for jj, atom_j in enumerate(neighbors):
                energies[atom_j] += 0.5 * pairwise_energies[jj]
                if 'forces' in properties:
                    forces[atom_j] += -pairwise_forces[jj]  # f_ji = - f_ij

        energy = energies.sum()

        return energy, forces, energies

    def batch_predict(self, data, over_write=False):

        energies = np.zeros(len(data))
        for i, atoms in enumerate(data):
            E, _, _ = self.get_model_properties(atoms, properties='energy')
            energies[i] = E

        if over_write:
            self.batch_assign(data, energies)

        return energies

    ####################################################################################################################
    # Training Methods:
    ####################################################################################################################    

    def train_model(self, training_data, energies):
        
        eps0 = self.get_array_parameters(self.epsilon_parameters)
        sigma0 = self.get_array_parameters(self.sigma_parameters)

        self.E_target = energies
        x0 = np.append(eps0, sigma0)

        print(self.loss_function(x0, *tuple(training_data)))
        xopt = fmin(self.loss_function, x0, tuple(training_data), maxiter=25)
        print(self.loss_function(xopt, *tuple(training_data)))

        eps_opt = xopt[0:self.num_interactions]
        sigma_opt = xopt[self.num_interactions:]
        self.update_epsilon_parameters(eps_opt)
        self.update_sigma_parameters(sigma_opt)

        self.set_ready_state(True)
    
    def loss_function(self, pars, *args):
        training_data = list(args)

        eps_arr = pars[0:self.num_interactions]
        sigma_arr = pars[self.num_interactions:]
        self.update_epsilon_parameters(eps_arr)
        self.update_sigma_parameters(sigma_arr)

        #E_target = np.array([atoms.get_potential_energy() for atoms in training_data])
        E_target = self.E_target
        E_predict = np.zeros_like(E_target)

        for i, atoms in enumerate(training_data):
            energy, _, _ = self.get_model_properties(atoms=atoms, properties=['energy'])
            E_predict[i] = energy

        error = np.mean((E_target - E_predict)**2)
        #error = np.mean((np.abs(E_target - E_predict)))

        return error

    ####################################################################################################################
    # Parameter Handling methods:
    ####################################################################################################################    

    def update_parameters(self, arr):
        """
        The supplied arr has length ntypes*(ntypes+1)/2
        """
        parameters = np.zeros((self.n_types, self.n_types))
        c = 0
        for i in range(self.n_types):
            for j in range(i, self.n_types):
                parameters[i, j] = arr[c]
                c += 1

        parameters = parameters + parameters.T - np.diag(np.diag(parameters))
        return parameters

    def update_epsilon_parameters(self, arr):
        self.epsilon_parameters = self.update_parameters(arr)

    def update_sigma_parameters(self, arr):
        self.sigma_parameters = self.update_parameters(arr)

    def get_array_parameters(self, parameters):
        return parameters[self.iidxs, self.jidxs]

    def set_training_data(self, data):
        self.training_data = data
    
    def get_training_data(self, data):
        return self.training_data
    
    def get_index_matrix(self, atoms):
        numbers = atoms.get_atomic_numbers()
        number_indexs = np.array([self.number_to_idx[n] for n in numbers])
        I, J = np.meshgrid(number_indexs, number_indexs)
        return I, J

    def get_sigma_matrix(self, I, J):
        return self.sigma_parameters[I, J]

    def get_epsilon_matrix(self, I, J):
        return self.epsilon_parameters[I, J]
