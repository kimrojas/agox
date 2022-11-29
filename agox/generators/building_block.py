from agox.generators.ABC_generator import GeneratorBaseClass
from ase.data import covalent_radii
from ase import Atom
import numpy as np

from ase.constraints import FixInternals

from copy import deepcopy

from scipy.spatial.distance import cdist

class BuildingBlockGenerator(GeneratorBaseClass):

    name = 'BuildingBlockGenerator'

    def __init__(self, building_blocks, N, apply_constraint=False, **kwargs):
        super().__init__(**kwargs)
        self.building_blocks = building_blocks
        self.N = np.array(N)

        for building_block in self.building_blocks:
            building_block.positions -= building_block.get_center_of_mass()

        self.apply_constraint = apply_constraint
        if apply_constraint:
            self.fix_internal_constraints = []
            for block in building_blocks:
                for constraint in block.constraints:
                    if isinstance(constraint, FixInternals):
                        self.fix_internal_constraints.append(constraint)
                    else:
                        raise NotImplementedError('Transferring constraints other than FixInternals from building blocks is not implemented.')
                        
    def get_candidates(self, sampler, environment):
        template = environment.get_template()
        numbers_list = environment.get_numbers()

        # if len(numbers_list) != len(np.repeat(self.building_block.get_atomic_numbers(), self.N)):
        #     self.writer('Building Blocks and N does not match environment settings!')
        #     self.writer('Will break now!')
        #     exit()

        len_of_template = len(template)

        candidate = template.copy()

        placed = np.zeros(len(self.building_blocks))
        count = 0
        bbs_used = []
        while np.sum(placed) < np.sum(self.N):
            #print(count, np.sum(placed)); count += 1
            build_succesful = True
            bb, bb_index = self.get_building_block(placed)
            
            if (placed == 0).all():
                suggested_position = self.get_box_vector()
            else:
                r_min = 1; r_max = 3.5
                radius = np.random.rand() * (r_max - r_min) + r_min
                displacement = self.get_displacement_vector(radius)
                xyz0 = candidate.positions[np.random.randint(low=0, high=len(candidate), size=1)].reshape(3)
                suggested_position = xyz0 + displacement

                if not self.check_confinement(suggested_position):
                    continue
            
            bb.positions += suggested_position

            # Checks confinement for all atoms in the building block at once.
            confinement_bool = self.check_positions_within_confinement(bb.positions)
            if not confinement_bool.all():
                continue
            
            # Checks that distances are not too bad.
            # This implementation does NOT care about PBCs.
            if np.sum(placed) > 0:
                distances = cdist(candidate.positions, bb.positions)
                if distances.min() < 1:
                    build_succesful = False
                if distances.min() > 1.75:
                    build_succesful = False

            if build_succesful:
                candidate += bb
                placed[bb_index] += 1
                bbs_used.append(bb_index)
            
        candidate = self.convert_to_candidate_object(candidate, template)
        candidate.add_meta_information('description', self.name)

        if self.apply_constraint:
            constraint, bonds, angles, dihedrals = self.get_fix_internal_constraint(bbs_used)
            candidate.set_constraint(constraint)

            return [candidate], bonds, angles, dihedrals

        return [candidate]

    def get_building_block(self, placed):

        # Get a building block:
        remaining = np.argwhere((self.N - placed) > 0).flatten()
        index = int(np.random.choice(remaining, size=1)[0])
        bb = self.building_blocks[index].copy()

        phi0, phi1, phi2 = np.random.uniform(0, 360, size=3)
        bb.rotate(phi2, (0,0,1))
        bb.rotate(phi0, (1,0,0))
        bb.rotate(phi1, (0,1,0))
        return bb, index

    def get_fix_internal_constraint(self, bbs_used):
        blocks_in_order = [self.building_blocks[index] for index in bbs_used]
        index_offsets = np.cumsum([0] + [len(bb) for bb in blocks_in_order])

        all_bonds = []
        all_angles = []
        all_dihedrals = []

        print(bbs_used)
        for bb_index, atom_index_offset in zip(bbs_used, index_offsets):
            print(atom_index_offset)

            base_constraint = deepcopy(self.fix_internal_constraints[bb_index])

            # Get the base constraint list:
            bonds = base_constraint.bonds.copy()
            angles = base_constraint.angles.copy()
            dihedrals = base_constraint.dihedrals.copy()

            # Update the indices:
            for value_index_list in [bonds, angles, dihedrals]:
                for value_index in value_index_list:
                    value_index[1] += atom_index_offset

            all_bonds += bonds
            all_angles += angles
            all_dihedrals += dihedrals

        return FixInternals(bonds=all_bonds, angles=all_angles, dihedrals=all_dihedrals), all_bonds, all_angles, all_dihedrals

