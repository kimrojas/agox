import pytest
import numpy as np
from agox.models import ModelGPR
from agox.generators import RandomGenerator

from environment_test import environment_and_dataset

def test_model(environment_and_dataset):
    environment, dataset = environment_and_dataset
    model = ModelGPR.default(environment, database=None)
    energies = np.array([atoms.get_potential_energy() for atoms in dataset])
    model.train_model(dataset, energies)

    # Test structures:
    generator = RandomGenerator(**environment.get_confinement())
    structures = [generator(None, environment)[0] for _ in range(10)]

    for atoms in structures:
        model.predict_energy(atoms)
        model.predict_forces(atoms)

        atoms.calc = model
        F = atoms.get_forces()
        E = atoms.get_potential_energy()


    parameters = model.get_model_parameters()
    model.set_model_parameters(parameters)




