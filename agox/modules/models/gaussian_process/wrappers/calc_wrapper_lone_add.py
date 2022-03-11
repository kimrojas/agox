from ase.calculators.calculator import Calculator, all_changes
from ase import Atom, Atoms
from ase.symbols import Symbols

import numpy as np

def get_wrapped_calculator(calc_to_type_to_wrap):

    class WrappedModelCalculator(calc_to_type_to_wrap):

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ever_trained = True
            self.feature_dict = {}

            self.implemented_properties.append('uncertainty')
            self.results['uncertainty'] = 0 # Because this wraps a accurate the default uncertainty is zero.

        def update(self):   
            pass

        def assign_from_ASLA(self, asla, main):
            #self._prepare_for_completing_grids(asla.builder, asla.grid.template)
            self.padding = asla.agent.padding

        def _put_missing_atoms_at_infinity(self, grid):
            """
            Doesn't actually put atoms at infinity. 
            """
            return grid

        def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):

            #if ['uncertainty'] in properties:
            self.results['uncertainty'] = 0

            if hasattr(atoms, 'get_feature') and 'forces' not in properties:
                key = hash(atoms.get_feature(padding=self.padding, local=False).tostring())
                if self.feature_dict.get(key, None) is not None:
                    self.results['energy'] = self.feature_dict.get(key, None)
                else:
                    super().calculate(atoms=atoms, properties=properties, system_changes=all_changes)

                    self.feature_dict[key] = self.results['energy']
            else:
                super().calculate(atoms=atoms, properties=properties, system_changes=all_changes)


    return WrappedModelCalculator
