from abc import ABC, abstractmethod
import awkward as ak
import numpy as np


class KernelBaseClass(ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hyperparameters = np.array([])
        self._hyperparameter_bounds = []


    def __call__(self, features_N, features_M=None):
        # print('memory usage',process.memory_info().rss)

        if features_M is None:
            if len(features_N.shape) == 3: # non-jagged!
                return self._get_global_kernel(features_N, features_N, symmetric=True)
            else: # jagged - make awkward
                jagged_features_N = [a for a in features_N] # ak.Array([a for a in features_N])
                return self._get_global_kernel(jagged_features_N, jagged_features_N, symmetric=True)
                
        else:
            if len(features_N.shape) == 3 and len(features_M.shape)==3: # non-jagged!
                return self._get_global_kernel(features_N, features_M)
            else: # jagged - make awkward
                jagged_features_N = [a for a in features_N] # ak.Array([a for a in features_N])
                jagged_features_M = [a for a in features_M] # ak.Array([a for a in features_M])
                return self._get_global_kernel(jagged_features_N, jagged_features_M)

        
    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def _set_hyperparameters(self, hyperparameters):
        pass

    @property
    def hyperparameters(self):
        return self._hyperparameters
    
    @hyperparameters.setter
    def hyperparameters(self, hyperparameters):
        self._set_hyperparameters(hyperparameters)
        self._hyperparameters = hyperparameters


    @property
    def hyperparameter_bounds(self):
        return self._hyperparameter_bounds
    
    @hyperparameter_bounds.setter
    def hyperparameter_bounds(self, bounds):
        self._hyperparameter_bounds = bounds
