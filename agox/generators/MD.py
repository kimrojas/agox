from ase.units import fs, kB
from ase.md.langevin import Langevin
from agox.generators.ABC_generator import GeneratorBaseClass
import numpy as np
from ase.constraints import FixAtoms, FixedLine, FixedPlane
from ase.io import read, write

class MDgenerator(GeneratorBaseClass):

    name = 'MDgenerator'

    def __init__(
            self,
            calculator,
            thermostat = Langevin,
            thermostat_kwargs = {'timestep':1.*fs, 'temperature_K':10, 'friction':0.05},
            temperature_program = [(500,10),(100,10)], 
            constraints=[],
            check_template = False,
            **kwargs):
        super().__init__(**kwargs)

        self.calculator = calculator # Calculator for MD simulation
        self.thermostat = thermostat # MD program used
        self.thermostat_kwargs = thermostat_kwargs # Settings for MD program
        self.temperature_program = temperature_program # (temp in kelvin, steps) for MD program if temperature is modifiable during simulation
        self.constraints = constraints # Constraints besides fixed template and 1D/2D constraints

        self.check_template = check_template # Check if template atoms moved during MD simulation


    def get_candidates(self, sampler, environment):
        candidate = sampler.get_random_member()

        # Happens if no candidates are part of the sample yet.
        if candidate is None:
            return [None]

        candidate.set_calculator(self.calculator)

        self.remove_constraints(candidate) # All constraints are removed from candidate before applying self.constraints to ensure only constraints set by user are present during MD simulation
        self.apply_constraints(candidate)
        self.molecular_dynamics(candidate)
        self.remove_constraints(candidate) # All constraints are removed after MD simulation to not interfere with other AGOX modules

        template = candidate.get_template()
        candidate = self.convert_to_candidate_object(candidate, template)
        candidate.add_meta_information('description', self.name)

        return [candidate]


    def molecular_dynamics(self, candidate):
        """ Runs the molecular dynamics simulation and applies/removes constraints accordingly """
        
        dyn = self.thermostat(candidate, **self.thermostat_kwargs)

        if self.check_template: 
            positions_before = candidate.template.positions

        for temp, steps in self.temperature_program:
            dyn.set_temperature(temperature_K=temp)
            dyn.run(steps)

        if self.check_template:
            positions_after = candidate.template.positions

            if np.array_equal(positions_before, positions_after):
                self.writer('Template positions were not altered by MD simulation')
                #print('Template positions were not altered by MD simulation')
            else:
                self.writer('Template positions were altered by MD simulation')
                #print('Template positions were altered by MD simulation')


    def apply_constraints(self, candidate):
        """ Applies constraints manually set and based on dimensionality of confinement cell """

        constraints = [] + self.constraints # Add any passed constraints immediately

        if self.dimensionality == 1 or self.dimensionality == 2: # Ensures movement within 1D or 2D confinement cell
            constraints.append(self.get_dimensionality_constraints(candidate))
        
        candidate.set_constraint(constraints)


    def get_dimensionality_constraints(self, candidate):
        """ Depending on the dimensionality this either sets fixed line or fixed plane.
        Similar to how rattle needs dimensionality specified to match with confinement cell"""

        template = candidate.get_template()
        n_template = len(template)
        n_total = len(candidate)

        if self.dimensionality == 1:
            return FixedLine(indices=np.arange(n_template, n_total), direction = [1,0,0]) # Generator assumes 1D search happens in X direction
        if self.dimensionality == 2:
            return FixedPlane(indices=np.arange(n_template, n_total), direction = [0,0,1]) # Generator assumes 2D search happens in XY-plane


    def remove_constraints(self, candidate):
        """ Removes all constraints from candidate """

        candidate.set_constraint([])
