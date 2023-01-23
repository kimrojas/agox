import os
import pickle
import numpy as np 
from agox.test.test_utils import compare_candidates
from agox.samplers import SamplerBaseClass
from agox.environments import Environment
from ase.io import read, write
from agox.test.test_utils import test_data_dicts, test_folder_path
from agox.candidates import StandardCandidate

def generator_test(generator, sampler, environment, expected_candidates, iterations=1, seed=1):
    np.random.seed(seed)

    list_of_list_of_candidates = []
    for iteration in range(iterations):
        list_of_list_of_candidates.append(generator(sampler, environment))

    if expected_candidates is not None:
        for list_of_candidates, list_of_expected in zip(list_of_list_of_candidates, expected_candidates):
            for candidate, expected_candidate in zip(list_of_candidates, list_of_expected):
                assert compare_candidates(candidate, expected_candidate)

    return list_of_list_of_candidates

def get_test_environment(path, remove):
    atoms = read(path, index=0)
    cell = atoms.get_cell()
    corner = np.array([0, 0, 0])
    numbers = atoms.get_atomic_numbers()[len(atoms)-remove:]

    template = atoms.copy()
    del template[len(template)-remove:len(template)]
    environment = Environment(template=template, numbers=numbers, confinement_cell=cell, 
            confinement_corner=corner)

    return environment

def get_test_data(path, environment):
    template = environment.get_template()
    atoms_data = read(path, index=':')
    candidate_data = [StandardCandidate.from_atoms(template, atoms) for atoms in atoms_data]
    return candidate_data

def get_test_sampler(data):

    class DummySampler(SamplerBaseClass):
        
        name = 'DummySampler'

        def __init__(self, data):
            self.sample = data

        def setup(self, *args, **kwargs):
            return None

    return DummySampler(data)

def get_name(generator, dataset_name, parameter_index):
    folder  = 'expected_outputs/'
    #name = folder + generator.name + dataset_name + parameter_index + '.pckl'
    name = f'{test_folder_path}generator_tests/{folder}{generator.name}_data{dataset_name}_parameter{parameter_index}.pckl'
    return name

def save_expected_data(data, generator, dataset_name, parameter_index):
    name = get_name(generator, dataset_name, parameter_index)
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_expected_data(generator, dataset_name, parameter_index):
    name = get_name(generator, dataset_name, parameter_index)
    with open(name, 'rb') as f:
        data = pickle.load(f)

    return data

def generator_testing(generator_class, test_data_dict, generator_args, base_kwargs, 
    update_kwargs, parameter_index, seed, test_mode=True):

    # Unpack data dictionary
    path = test_data_dict['path']
    remove = test_data_dict['remove']
    dataset_name = test_data_dict['name']

    # Setup required things:
    environment = get_test_environment(path, remove)
    data = get_test_data(path, environment)
    sampler = get_test_sampler(data)

    # Create the generator
    generator_kwargs = base_kwargs.copy()
    generator_kwargs.update(update_kwargs)
    generator = generator_class(*generator_args, **environment.get_confinement(), **generator_kwargs)

    # Load expected data:
    if test_mode:
        expected_data = load_expected_data(generator, dataset_name, parameter_index)
    else:
        expected_data = None
    
    # Run the actual test. 
    output = generator_test(generator, sampler, environment, expected_data, iterations=5, seed=seed)

    if not test_mode:
        save_expected_data(output, generator, dataset_name, parameter_index)

    return output



