import pickle
import numpy as np 
from agox.test.test_utils import compare_candidates
from agox.samplers import SamplerBaseClass
from agox.environments import Environment

def test_generator(generator, sampler, environment, expected_candidates, seed=1):
    np.random.seed(seed)

    candidates = generator(sampler, environment)

    if expected_candidates is not None:
        for candidate, expected_candidates in zip(candidates, expected_candidates):
            assert compare_candidates(candidate)

    return candidates

def get_test_environment(atoms, remove):
    cell = atoms.get_cell()
    corner = np.array([0, 0, 0])
    numbers = atoms.get_atomic_numbers()[len(atoms)-remove:]

    template = atoms.copy()
    del template[len(template)-remove:len(template)]
    environment = Environment(template=template, numbers=numbers, confinement_cell=cell, 
            confinement_corner=corner)

    return environment

def get_test_sampler(data, environment):

    class DummySampler:

        def __init__(self, data):
            self.sample = data

        def setup(self, *args, **kwargs):
            return None

    return DummySampler(data)

def save_expected_data(generator, data):

    folder  = 'expected_outputs/'
    name = folder + generator.name + str(data[0].symbols.formula)  + '.pckl'
    
    with open(name, 'wb') as f:
        pickle.dump(data, f)

