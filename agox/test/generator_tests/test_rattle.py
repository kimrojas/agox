import pytest
import numpy as np
from agox.generators import RattleGenerator
from agox.test.generator_tests.generator_utils import generator_test, get_test_environment, get_test_sampler, save_expected_data, load_expected_data, get_test_data
from agox.test.test_utils import test_data_dicts

seed = 1
generator_args = []
generator_base_kwargs = {'c1':0.75, 'c2':1.25, 'dimensionality':3}
generator_class = RattleGenerator

list_of_other_kwargs = [
    {},
    {'n_rattle':3, 'rattle_amplitude':3},
    {'n_rattle':5, 'rattle_amplitude':5}
    ]

for index, dictionary in enumerate(list_of_other_kwargs):
    list_of_other_kwargs[index] = (dictionary, index)

@pytest.fixture(params=list_of_other_kwargs)
def other_kwargs(request):
    return request.param

@pytest.mark.parametrize('test_data_dict', test_data_dicts)
def test_random(test_data_dict, other_kwargs):

    parameter_index = other_kwargs[1]
    other_kwargs = other_kwargs[0]

    path = test_data_dict['path']
    remove = test_data_dict['remove']
    dataset_name = test_data_dict['name']

    environment = get_test_environment(path, remove)
    data = get_test_data(path, environment)
    sampler = get_test_sampler(data)

    generator_kwargs = generator_base_kwargs.copy()
    generator_kwargs.update(other_kwargs)

    generator = generator_class(*generator_args, **environment.get_confinement(), **generator_kwargs)

    expected_data = load_expected_data(generator, dataset_name, parameter_index)
    
    generator_test(generator, sampler, environment, expected_data, iterations=5, seed=seed)

if __name__ == '__main__':

    for test_data_dict in test_data_dicts:

        for other_kwargs, parameter_index in list_of_other_kwargs:

            dataset_name = test_data_dict['name']
            path = test_data_dict['path']
            remove = test_data_dict['remove']

            environment = get_test_environment(path, remove)
            data = get_test_data(path, environment)
            sampler = get_test_sampler(data)

            generator_kwargs = generator_base_kwargs.copy()
            generator_kwargs.update(other_kwargs)
            generator = generator_class(*generator_args, **environment.get_confinement(), **generator_kwargs)

            output = generator_test(generator, sampler, environment, None, iterations=5, seed=seed)

            save_expected_data(output, generator, dataset_name, parameter_index)

        
