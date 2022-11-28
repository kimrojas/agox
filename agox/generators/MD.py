from ase import units
from ase.md.langevin import Langevin
from agox.generators.ABC_generator import GeneratorBaseClass
import numpy as np
from ase.constraints import FixAtoms, FixedLine, FixedPlane

class MDgeneratorLangevin(GeneratorBaseClass):

    name = 'MDgeneratorLangevin'

    def __init__(self, calculator, temperatures=[2000], step_lengths = [50], timestep=2, friction = 0.1,
                 choose_lowest_energy = False, constraints = None, **kwargs):
        super().__init__(**kwargs)

        self.calculator = calculator # Calculator used in MD simulation
        self.temperatures = temperatures # List of temperatures to be used during MD simulation
        self.step_lengths = step_lengths # List of amount of steps taken at each temperature during MD simulation
        self.timestep = timestep * units.fs  # Timestep given in units of femtoseconds
        self.friction = friction # Friction coefficient coupling atoms to heat bath

        self.choose_lowest_energy = choose_lowest_energy # Return structure with lowest energy during MD simulation

        self.constraints = constraints # Constraints for non-template atoms


    def get_candidates(self, sampler, environment):
        candidate = sampler.get_random_member()

        # Happens if no candidates are part of the sample yet.
        if candidate is None:
            return [None]

        template = candidate.get_template()

        candidate.set_calculator(self.calculator)
        self.molecular_dynamics(candidate)

        candidate = self.convert_to_candidate_object(candidate, template)
        candidate.add_meta_information('description', self.name)

        return [candidate]


    def molecular_dynamics(self, candidate):
        """ Does the actual molecule dynamics until specified steps is reached"""

        energies = []
        old_positions = []

        # Template is fixed during MD
        fixed_template_constraint = self.get_fixed_template_constraint(candidate)
        candidate.set_constraint(fixed_template_constraint)

        # Apply constraint to candidate depending on dimensionality or pre-set constraint
        if self.constraints == None:
            dimensionality_constraint = self.get_dimensionality_constraints(candidate)
            candidate.set_constraint(dimensionality_constraint)
        else:
            candidate.set_constraint(self.constraints)

        dynamics = Langevin(candidate, timestep=self.timestep, temperature_K = 100, friction = self.friction) # temperature is set but changed later
        for i in range(len(self.step_lengths)):
            dynamics.set_temperature(temperature_K = self.temperatures[i])
            for _ in range(self.step_lengths[i]):
                dynamics.run(1)
                if self.choose_lowest_energy:
                    energies.append(candidate.get_potential_energy())
                    old_positions.append(candidate.positions.copy())
        
        if self.choose_lowest_energy:
            lowest_energy_index = self.get_lowest_energy_structure_from_MD_simulation(energies)
            candidate.positions = old_positions[lowest_energy_index]
            
        # Constraint is removed after MD simulation to make sure it does not interfere with other parts of AGOX
        candidate.set_constraint([])


    def get_fixed_template_constraint(self, candidate):
        return FixAtoms(indices=np.arange(len(candidate.get_template())))


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


    def get_lowest_energy_structure_from_MD_simulation(self, energies):
        """ Takes a list of energies for each step of a MD simulation and returns index with lowest energy """

        lowest_energy_index = np.argmin(energies)
        return lowest_energy_index

