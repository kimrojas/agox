from matplotlib import use
import numpy as np
from ase.data import covalent_radii
from .model_ABC import ModelBaseClass
from ase.calculators.calculator import Calculator, all_changes

from scipy.optimize import fmin, minimize

class MorseModel(ModelBaseClass):

    implemented_properties = ['energy', 'forces']

    name = 'Morse'

    def __init__(self, use_cutoff=False, **kwargs):
        super().__init__(**kwargs)
        self.use_cutoff = use_cutoff
        self.set_ready_state(True)

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
        self.r0_parameters = sigma_parameters + sigma_parameters.T - np.diag(np.diag(sigma_parameters))        
        self.epsilon_parameters = epsilon_parameters + epsilon_parameters.T - np.diag(np.diag(epsilon_parameters))
        self.rho_parameters = np.ones((self.n_types, self.n_types))

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        energy, forces = self.get_model_properties(atoms=atoms, properties=properties)

        self.results['energy'] = energy
        self.results['forces'] = forces

    def get_model_properties(self, atoms=None, properties=['energy']):
        # epsilon = self.parameters.epsilon
        # rho0 = self.parameters.rho0
        # r0 = self.parameters.r0
        I, J = self.get_index_matrix(atoms)
        epsilon_matrix = self.epsilon_parameters[I, J]
        r0_matrix = self.r0_parameters[I, J]
        rho_matrix = self.rho_parameters[I, J]

        positions = atoms.get_positions()
        energy = 0.0
        forces = np.zeros((len(atoms), 3))
        #preF = 2 * epsilon * rho0 / r0
        for i1, p1 in enumerate(positions):
            for i2, p2 in enumerate(positions[:i1]):
                rho0 = rho_matrix[i1, i2]
                epsilon = epsilon_matrix[i1, i2]
                r0 = r0_matrix[i1, i2]
                preF = 2 * epsilon * rho0 / r0
                diff = p2 - p1
                r = np.sqrt(np.dot(diff, diff))
                if self.use_cutoff:
                    R = r0*1.5#(covalent_radii[atoms[i1].number] + covalent_radii[atoms[i2].number])*1.30
                    if r > R:
                        r = r + ((r/R)**2 - 1) * 5
                        #r = r + (R - r)
                        #continue

                expf = np.exp(rho0 * (1.0 - r / r0))
                F = preF * expf * (expf - 1) * diff / r
                forces[i1] -= F
                forces[i2] += F
                energy += epsilon * expf * (expf - 2)
                

        return energy, forces

    def batch_predict(self, data, over_write=False):
        energies = np.zeros(len(data))
        for i, atoms in enumerate(data):
            E, _, = self.get_model_properties(atoms, properties='energy')
            energies[i] = E

        if over_write:
            self.batch_assign(data, energies)

        return energies

    def get_index_matrix(self, atoms):
        numbers = atoms.get_atomic_numbers()
        number_indexs = np.array([self.number_to_idx[n] for n in numbers])
        I, J = np.meshgrid(number_indexs, number_indexs)
        return I, J

    def train_model(self, training_data, energies):
        #return super().train_model(training_data, energies)
        pass

    ####################################################################################################################
    # Fitting Methods:
    ####################################################################################################################

    def autofit(self, training_data, energies, r, reference_dict={}):

        numbers = training_data[0].numbers
        
        # Need these indexs:
        Zi = self.number_to_idx[numbers[0]]
        Zj = self.number_to_idx[numbers[1]]

        # r0 and epsilon are assigned:
        r0 = r[np.argmin(energies)]
        reference = (reference_dict.get(numbers[0], 0)+reference_dict.get(numbers[1], 0))
        epsilon = np.abs(np.min(energies) - reference)

        # Set these parameters:
        self.epsilon_parameters[Zi, Zj] = epsilon
        self.epsilon_parameters[Zj, Zi] = epsilon
        self.r0_parameters[Zi, Zj] = r0
        self.r0_parameters[Zj, Zi] = r0

        # Can fit rho0
        args = (training_data, Zi, Zj, energies-reference)
        rho0 = 1.4
        rho_fit = fmin(self.rho_fitting_loss, rho0, args, maxiter=100)

        self.rho_parameters[Zi, Zj] = rho_fit
        self.rho_parameters[Zj, Zi] = rho_fit

        print('Parameters found:')
        for name, value in zip(['epsilon', 'r0', 'rho'], [epsilon, r0, rho_fit]):
            print('{} = {}'.format(name, value))

    def rho_fitting_loss(self, rho, *args):
        
        # Get stuff from args:
        training_data = args[0]
        Zi = args[1]
        Zj = args[2]
        E_target = args[3]

        # Set incoming parameters:
        self.rho_parameters[Zi, Zj] = rho
        self.rho_parameters[Zj, Zi] = rho

        # Calculate MSE
        E_predict = np.zeros_like(E_target)
        for i, atoms in enumerate(training_data):
            energy, _, = self.get_model_properties(atoms=atoms, properties=['energy'])
            E_predict[i] = np.min([energy, 0])

        error = np.mean((E_target - E_predict)**2)

        return error

    def save_parameters(self, prefix=''):
        np.save(prefix+'epsilon.npy', self.epsilon_parameters)
        np.save(prefix+'r0.npy', self.r0_parameters)
        np.save(prefix+'rho.npy', self.rho_parameters)
        
    def load_parameters(self, directory='', prefix=''):
        self.epsilon_parameters = np.load(directory+prefix+'epsilon.npy')
        self.r0_parameters = np.load(directory+prefix+'r0.npy')
        self.rho_parameters = np.load(directory+prefix+'rho.npy')
        print('Succesfully loaded model_parameters')
        