import pytest
import numpy as np
from agox.generators import SteepestDescentGenerator
from agox.test.generator_tests.generator_utils import generator_testing
from agox.test.test_utils import test_data_dicts

seed = 1
generator_args = []
generator_base_kwargs = {'c1':0.75, 'c2':1.25, 'dimensionality':3}
generator_class = SteepestDescentGenerator

list_of_other_kwargs = [
    {},
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

    generator_testing(generator_class, test_data_dict, generator_args, 
        generator_base_kwargs, other_kwargs, parameter_index, seed=seed)

if __name__ == '__main__':

    for test_data_dict in test_data_dicts:

        for other_kwargs, parameter_index in list_of_other_kwargs:

            output = generator_testing(generator_class, test_data_dict, generator_args, 
                generator_base_kwargs, other_kwargs, parameter_index, seed=seed, test_mode=False)

        
