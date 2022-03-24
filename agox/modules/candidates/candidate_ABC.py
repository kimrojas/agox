from abc import ABC, abstractmethod
from ase import Atoms
import numpy as np

from copy import deepcopy
from ase.calculators.singlepoint import SinglePointCalculator

class CandidateBaseClass(Atoms, ABC):

    def __init__(self, template=None, template_indices=None, **kwargs):
        super().__init__(**kwargs) # This means all Atoms-related stuff gets set. 
        self.meta_information = dict()

        # Template stuff:        
        if template_indices is not None:
            self.template_indices = template_indices.astype(int)
            self.template = self.get_template()
        elif template is not None:
            self.template = template
            self.template_indices = np.arange(len(template))
        else:
            print('You have not supplied a template, using an empty atoms object without PBC and no specified cell.')
            self.template = Atoms(pbc=self.pbc)            
            self.template_indices = np.arange(0)
        
        self.set_pbc(self.template.get_pbc()) # Inherit PBC's from template.
        
        self.postprocess_immunity = False

        # This will happen eventually when work starts on using multiple templates.
        # But the check doesnt work as intended at the moment.
        # if len(template) > 0:            
        #     assert (self.positions[:len(template)] == template.positions).all(), 'Template and positions do not match'

    def add_meta_information(self, name, value):
        self.meta_information[name] = value

    def get_meta_information(self, name):
        try:
            return self.meta_information.get(name, None).copy()
        except AttributeError: 
            # This catches for example 'int' that dont have a copy method. 
            # Ints won't change in-place, but it perhaps custom classes will. 
            return self.meta_information.get(name, None)

    def get_meta_information_no_copy(self, name):
        return self.meta_information.get(name, None)

    def has_meta_information(self, name):
        return name in self.meta_information.keys()

    def pop_meta_information(self, name):
        return self.meta_information.pop(name, None)

    def get_template(self):
        return Atoms(numbers=self.numbers[self.template_indices], positions=self.positions[self.template_indices], cell=self.cell, pbc=self.pbc)

    def copy(self):
        """
        Return a copy of candidate object. 

        Not sure if template needs to be copied, but will do it to be safe.
        """
        candidate = self.__class__(template=self.template.copy(), cell=self.cell, pbc=self.pbc, info=self.info,
                               celldisp=self._celldisp.copy())
        candidate.meta_information = self.meta_information.copy()
        
        candidate.arrays = {}
        for name, a in self.arrays.items():
            candidate.arrays[name] = a.copy()
        # Not copying constraints because those should be handled by the environment/postprocessors that use them.             
        #atoms.constraints = copy.deepcopy(self.constraints)
        return candidate

    def copy_calculator_to(self, atoms):
        '''
        Copy current calculator and attach to the atoms object
        '''
        if self.calc is not None and 'energy' in self.calc.results:
            if 'forces' in self.calc.results:
                calc = SinglePointCalculator(atoms, energy=self.calc.results['energy'],
                                             forces=self.calc.results['forces'])
            else:
                calc = SinglePointCalculator(atoms, energy=self.calc.results['energy'])
            atoms.set_calculator(calc)

    def get_uncertainty(self):
        if 'uncertainty' in self.calc.implemented_properties:
            return self.calc.get_property('uncertainty')
        else:
            #print('Calculator {} does not implement uncertainty - Default return: 0')
            return 0.

    def set_postprocess_immunity(self, state):
        self.postprocess_immunity = state

    def get_postprocess_immunity(self):
        return self.postprocess_immunity

    def get_property(self, key):
        return self.calc.get_property(key)

    def get_template_indices(self):
        return self.template_indices

    def get_optimize_indices(self):
        return np.arange(len(self.template), self.get_global_number_of_atoms())

    def get_center_of_geometry(self, all_atoms=False):
        if all_atoms:
            return np.mean(self.positions, axis=0).reshape(1, 3)
        else:
            return np.mean(self.positions[self.get_optimize_indices()], axis=0).reshape(1, 3)