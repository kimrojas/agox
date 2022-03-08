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

        def update(self):   
            pass

        def assign_from_ASLA(self, asla, main):
            #self._prepare_for_completing_grids(asla.builder, asla.grid.template)
            self._prepare_for_completing_grids_org(asla.builder, asla.grid.template)
            self.padding = asla.agent.padding

        def get_incomplete_energy(self, grid):
            grid.set_calculator(self)
            return grid.get_potential_energy()

        def _prepare_for_completing_grids(self, builder, template):
            self.atom_types = builder.atom_types
            
            # Get atomic energy of each species
            self.energy_dict = {}
            for t in self.atom_types:
                temp_atoms = Atoms(numbers=[t], positions=np.array([[7.5, 7.5, 7.5]]), cell=np.eye(3)*15)
                temp_atoms.set_calculator(self)
                self.energy_dict[t] = temp_atoms.get_potential_energy()

            # Number of atoms of each species in completed structure:
            self.complete_build_dict = {}            

            for t in self.atom_types:
                self.complete_build_dict[t] = np.sum(template.numbers == t) + np.sum(np.array(builder.numbers) == t)
        
        def _prepare_for_completing_grids_org(self, builder, template):
            self.atom_types = builder.atom_types
            self.atoms_this_type_in_complete_structure = {}
            N_total = 0
            for t in self.atom_types:
                self.atoms_this_type_in_complete_structure[t] = \
                                    sum(template.numbers==t) + sum([1 for n in builder.numbers if n==t])
                N_total += self.atoms_this_type_in_complete_structure[t]
            self.N_total = N_total

        def get_missing_atoms(self, grid):
            missing_atoms = dict()
            for t in self.atom_types:
                missing_atoms[t] = self.complete_build_dict[t] - np.sum(grid.numbers == t)  
            return missing_atoms

        def get_simulated_energy(self, grid):
            return self.get_simulated_energy_V2(grid)

        def get_simulated_energy_V1(self, grid):            
            # Get the energy of current structure:
            grid.set_calculator(self)
            E = grid.get_potential_energy()

            missing_atoms = self.get_missing_atoms(grid)
            for t in missing_atoms.keys():
                E += missing_atoms[t] * self.energy_dict[t]

            return E

        def get_simulated_energy_V2(self, grid):
            # Get the energy of current structure:
            grid.set_calculator(self)
            E = grid.get_potential_energy()
            return E

        # def _put_missing_atoms_at_infinity(self, grid):
        #     dx = 100
        #     x = dx
        #     for t in self.atom_types:
        #         while sum(grid.numbers==t) < self.atoms_this_type_in_complete_structure[t]:
        #             grid.extend(Atom(t,[x,0,0]))
        #             x += dx
        #     return grid

        def _put_missing_atoms_at_infinity(self, grid):
            return grid

        def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
            key = hash(atoms.get_feature(padding=self.padding, local=False).tostring())
            if self.feature_dict.get(key, None) is not None:
                self.results['energy'] = self.feature_dict.get(key, None)
            else:
                super().calculate(atoms=atoms, properties=properties, system_changes=all_changes)
                self.feature_dict[key] = self.results['energy']

    return WrappedModelCalculator
