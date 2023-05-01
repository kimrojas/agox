from agox.generators.ABC_generator import GeneratorBaseClass
import numpy as np
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from scipy.optimize import minimize
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS

class ComplementaryEnergyGenerator(GeneratorBaseClass):

    name = 'ComplementaryEnergyGenerator'

    def __init__(self,
                calculator,
                descriptor,
                attractor_method,
                move_all = 1, 
                mover_indices = None,
                **kwargs):

        super().__init__(**kwargs)
        self.calc = calculator
        self.descriptor = descriptor
        self.attractor_method = attractor_method
        self.move_all = move_all
        self.mover_indices = mover_indices

    def get_candidates(self, sampler, environment):
        candidate = sampler.get_random_member()

        # Happens if no candidates are part of the sample yet. 
        if candidate is None:
            return [None]
        
        template = candidate.get_template()
        n_template = len(template)

        # Set attractors
        attractors = self.attractor_method.get_attractors(candidate)

        # Determine which atoms to move if no indices has been supplied
        if self.mover_indices is None:
            if self.move_all == 1:
                mover_indices = range(n_template, len(candidate))
            else:
                features = self.descriptor.get_local_features(candidate)
                n_movers = np.random.randint(1, len(candidate) - n_template)
                local_ce = self.calc.get_local_ce_energy(features, attractors)
                largest_local_ce = local_ce[n_template:] # sort out template
                largest_local_ce_indices = np.argsort(largest_local_ce)[::-1]
                mover_indices = largest_local_ce_indices[:n_movers] + n_template
            self.mover_indices = mover_indices

        # Set mover indices and attractors for calculator
        self.calc.mover_indices = self.mover_indices
        self.calc.attractors = attractors

        # Set calculator
        candidate.calc = self.calc

        def objective_func(pos):
            candidate.positions[self.mover_indices] = np.reshape(pos,[len(self.mover_indices),3])
            ce = candidate.get_potential_energy(apply_constraint = 0)
            grad = - candidate.get_forces(apply_constraint = 0)[self.mover_indices]
            return ce, grad.flatten()

        pos = candidate.positions[self.mover_indices].flatten()
        pos = minimize(objective_func, pos, method = 'BFGS', jac = True, options={'maxiter':75, 'disp':False}).x
        suggested_positions = np.reshape(pos,[len(self.mover_indices), 3])

        for index in range(len(self.mover_indices)):
            i = self.mover_indices[index]
            for _ in range(100):
                if _ == 0:
                    radius = 0
                else:
                    radius = 0.5 * np.random.rand()**(1/self.dimensionality)
                displacement = self.get_displacement_vector(radius)
                suggested_position = suggested_positions[index] + displacement

                # Check confinement limits:
                if not self.check_confinement(suggested_position):
                    continue

                # Check that suggested_position is not too close/far to/from other atoms
                # Skips the atom it self. 
                if self.check_new_position(candidate, suggested_position, candidate[i].number, skipped_indices=[i]):
                    candidate[i].position = suggested_position 
                    break

        candidate = self.convert_to_candidate_object(candidate, template)
        candidate.add_meta_information('description', self.name)
        return [candidate]

    @classmethod
    def default(cls, species, database, move_all = 1, mover_indices = None, attractors_from_template = False, predefined_attractors = None, **kwargs):

        from agox.models.descriptors.exponential_density import ExponentialDensity
        from agox.generators.complementary_energy.ce_calculators import ComplementaryEnergyDistanceCalculator
        from agox.generators.complementary_energy.attractor_methods.ce_attractors_current_structure import AttractorCurrentStructure

        lambs = [0.5, 1, 1.5]
        rc = 10.
        descriptor = ExponentialDensity(species, lambs = lambs, rc = rc)
        ce_calc = ComplementaryEnergyDistanceCalculator(descriptor = descriptor)
        ce_attractors = AttractorCurrentStructure(descriptor = descriptor, attractors_from_template = attractors_from_template, predefined_attractors = predefined_attractors)
        ce_attractors.attach(database)

        return cls(calculator = ce_calc, descriptor = descriptor, attractor_method = ce_attractors, **kwargs)

