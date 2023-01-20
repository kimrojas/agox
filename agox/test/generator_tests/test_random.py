import numpy as np
from agox.generators import RandomGenerator
from agox.test.generator_tests.generator_utils import test_generator, get_test_environment, get_test_sampler, save_expected_data
from ase import Atoms


if __name__ == '__main__':

    atoms = Atoms('H10', cell=np.eye(3))

    environment = get_test_environment(atoms, 5)
    sampler = get_test_sampler([atoms], environment)

    generator = RandomGenerator(**environment.get_confinement())

    output = test_generator(generator, sampler, environment, None)

    save_expected_data(generator, output)

    
