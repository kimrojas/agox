import numpy as np
from agox.modules.models.model_ABC import ModelBaseClass
from qml.representations import generate_coulomb_matrix
from qml.kernels import gaussian_kernel
from qml.math import cho_solve

class ModelQML(ModelBaseClass):

    implemented_properties = ['energy']

    name = 'QML'

    def __init__(self, max_num_atoms, representation=None, sigma=4000, **kwargs):
        super().__init__(**kwargs)
        self.representation = generate_coulomb_matrix if representation is None else representation
        self.sigma = sigma
        self.max_num_atoms = max_num_atoms
        
    def train(self, training_data):
        """
        Training data is list of atoms-objects.
        """
        self.X = np.array([self.representation(atoms.numbers, atoms.positions, size=self.max_num_atoms) for atoms in training_data])
        y = np.array([atoms.get_potential_energy() for atoms in training_data])

        K = gaussian_kernel(self.X, self.X, self.sigma)
        K[np.diag_indices_from(K)] += 1e-8

        self.alpha = cho_solve(K, y)

    def calculate(self, atoms=None, properties=['energy'], system_changes=None):
        x = self.representation(atoms.numbers, atoms.positions, size=self.max_num_atoms).reshape(1, -1)
        K = gaussian_kernel(x, self.X, self.sigma)
    
        E = np.dot(K, self.alpha)
        self.results['energy'] = E


if __name__ == '__main__':

    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    a = Atoms('H2', positions=[[1, 0, 0], [0, 0, 1]])
    calc = SinglePointCalculator(a, energy=10)
    a.set_calculator(calc)
    b = Atoms('H2', positions=[[0.9, 0, 0], [0, 0, 0.9]])
    calc = SinglePointCalculator(b, energy=9)
    b.set_calculator(calc)


    model = ModelQML(4)

    model.train([a, b])

    c = Atoms('H', positions=[[0.9, 0, 0]])
    c.set_calculator(model)

    E = c.get_potential_energy()
    print(E)
