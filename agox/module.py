from abc import ABC, abstractmethod
import functools
from uuid import uuid4
from copy import copy

class Module:

    dynamic_attributes = [] # Attributes that Ray make sure are updated on actors. 

    kwargs = ['surname']

    def __init__(self, use_cache=False, surname=''):
        self.surname = surname
        
        self.use_cache = use_cache
        self.cache_key = str(uuid4())

        self.tracked_attributes = [] # Attributes that the logger will track.
    
    def get_dynamic_attributes(self):
        return {key:self.__dict__[key] for key in self.dynamic_attributes}

    def add_dynamic_attribute(self, attribute_name):
        assert attribute_name in self.__dict__.keys()
        self.dynamic_attributes.append(attribute_name)

    def remove_dynamic_attribute(self, attribute_name):
        assert attribute_name in self.__dict__.keys()
        assert attribute_name in self.dynamic_attributes
        del self.dynamic_attributes[self.dynamic_attributes.index(attribute_name)]
    
    def get_tracked_attributes(self):
        attributes = dict()
        for key in self.tracked_attributes:
            try:
                attributes[key] = self.__dict__[key]
            except: # To avoid crashing if the attribute is not yet defined.
                attributes[key] = None
        return attributes
    
    def add_tracked_attribute(self, attribute_name):
        self.tracked_attributes.append(attribute_name)

    @property
    @abstractmethod
    def name(self):
        return NotImplementedError

    @property
    def dynamic_state(self):
        state = len(self.dynamic_attributes) > 0
        return state

    @property
    def __name__(self):
        """
        Defines the name.
        """
        if len(self.dynamic_attributes) > 0:
            last = 'Dynamic'
        else:
            last = ''
        return self.name + self.surname + last

    def find_submodules(self, in_key=None, only_dynamic=False):
        if in_key is None:
            in_key = []

        submodules = {}
        for key, value in self.__dict__.items():
            if issubclass(value.__class__, Module):
                key = in_key + [key]
                if only_dynamic:
                    if value.dynamic_state:                
                        submodules[tuple(key)] = value
                else:
                    submodules[tuple(key)] = value
                submodules.update(value.find_submodules(in_key=key, only_dynamic=only_dynamic))

        return submodules

    def set_for_submodule(self, submodule_keys, value):
        reference = self
        for key in submodule_keys[0:-1]:
            reference = self.__dict__[key]        
        reference.__dict__[submodule_keys[-1]] = value
        

    @classmethod
    def reset_cache_key(clc, func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            self.cache_key = str(uuid4())            
            return func(self, *args, **kwargs)
        return wrapper
        
