import pytest
import numpy as np
from agox.models.local_GPR.LSGPR_CUR import LSGPRModelCUR
from agox.models.descriptors.soap import SOAP

from agox.test.model_tests.model_utils import model_tester
from agox.test.test_utils import get_test_data, get_test_environment, load_expected_data, save_expected_data, label_dict_list
from agox.test.test_utils import test_data_dicts, get_name

from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

kernel = C(1)*RBF(length_scale=20)

model_class = LSGPRModelCUR
model_maker = LSGPRModelCUR
model_base_args = []
model_base_kwargs = {}

model_update_kwargs = [
    {'kernel':kernel, 'database':None},
]

model_update_kwargs = label_dict_list(model_update_kwargs)

@pytest.fixture(params=model_update_kwargs)
def update_kwargs(request):
    return request.param

@pytest.mark.parametrize('test_data_dict', test_data_dicts)
def test_model(test_data_dict, update_kwargs):

    update_kwargs, parameter_index  = update_kwargs

    path = test_data_dict['path']
    remove = test_data_dict['remove']
    dataset_name = test_data_dict['name']
    parameter_index = 0

    environment = get_test_environment(path, remove)
    data = get_test_data(path, environment)

    # Extra stuff:
    descriptor = SOAP(environment.get_all_species(), periodic=environment.get_template().pbc.any())
    update_kwargs['descriptor'] = descriptor

    # Slightly complicated way of building input args & kwargs:
    if 'environment' in update_kwargs.keys():
        update_kwargs['environment'] = environment

    model_kwargs = model_base_kwargs.copy()
    model_kwargs.update(update_kwargs)

    subfolder = 'model_tests/'
    module_name = model_class.name
    name = get_name(module_name, subfolder, dataset_name, parameter_index)
    expected_data = load_expected_data(name) 

    output = model_tester(model_maker, model_base_args, model_kwargs, data, test_mode=True, expected_energies=expected_data)

if __name__ == '__main__':

    test_data_dict = test_data_dicts[0]
    for test_data_dict in test_data_dicts:
        for update_kwargs, parameter_index in model_update_kwargs:

            path = test_data_dict['path']
            remove = test_data_dict['remove']
            dataset_name = test_data_dict['name']
            parameter_index = 0

            environment = get_test_environment(path, remove)
            data = get_test_data(path, environment)

            descriptor = SOAP(environment.get_all_species(), periodic=environment.get_template().pbc.any())
            update_kwargs['descriptor'] = descriptor

            # Slightly complicated way of building input args & kwargs:
            if 'environment' in update_kwargs.keys():
                update_kwargs['environment'] = environment

            model_kwargs = model_base_kwargs.copy()
            model_kwargs.update(update_kwargs)    

            subfolder = 'model_tests/'
            module_name = model_class.name
            name = get_name(module_name, subfolder, dataset_name, parameter_index)
            
            output = model_tester(model_maker, model_base_args, model_kwargs, data, test_mode=False)
            save_expected_data(name, output)










