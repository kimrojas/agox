from agox.generators.ABC_generator import GeneratorBaseClass
import numpy as np
from ase import Atoms
from ase.data import covalent_radii

from scipy.spatial.distance import cdist
from ase.constraints import FixInternals
from copy import deepcopy


class BlockGeneratorBaseClass(GeneratorBaseClass):

    def __init__(self, building_blocks, N_blocks, apply_constraint=False, **kwargs):
        super().__init__(**kwargs)

        self.building_blocks = building_blocks
        self.N_blocks = np.array(N_blocks)

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

    def extract_block(self, candidate, block_indices):
        positions = candidate.positions[block_indices]
        numbers = candidate.numbers[block_indices]
        cell = candidate.get_cell()

        atoms = Atoms(numbers=numbers, positions=positions, cell=cell)

        return atoms

    def block_move(self, candidate, block, internal_indices):
        candidate.positions[internal_indices] = block.positions

    def compare_environment_and_blocks(self, environment):
        numbers_list = environment.get_numbers()

        if len(numbers_list) != len(np.repeat(self.building_block.get_atomic_numbers(), self.N)):
            self.writer('Building Blocks and N does not match environment settings!')
            self.writer('Will break now!')
            exit()

    def check_distances(self, candidate, block, internal_indices=None, present=True):
        if len(candidate) == 0:
            return True

        # Find the distances of relevant atoms - we mask out the block 
        # itself if it is already present in the candidate object. 
        mask = np.ones(len(candidate)).astype(bool)
        if present:
            mask[internal_indices] = False
        D = cdist(block.positions, candidate.positions[mask, :])

        # We want to figure out if these are within allowed distances.
        # We make a matrix of the sum of covalent_radii such that 
        # cv_matrix[0, 0] is the sum of covalent radii between a 
        # block atom and a candidate atom. 
        N_block = len(block)
        N_candidate = np.count_nonzero(mask)
        block_numbers = np.repeat(block.numbers[:, np.newaxis], axis=1, repeats=N_candidate)
        candidate_numbers = np.repeat(candidate.numbers[mask][np.newaxis, :], axis=0, repeats=N_block)
        block_cv = covalent_radii[block_numbers]
        candidate_cv = covalent_radii[candidate_numbers]
        cv = block_cv + candidate_cv

        # First we check thhat all distances are over c1 * cv
        min_check = (self.c1 * cv < D).all()
        # Then check that ANY distance is smaller than c2 * cv
        max_check = (self.c2 * cv > D).any()
        # Combined this esnures that we at least have ONE bond of appropriate length 
        # and that NONE are too short. 
        return min_check * max_check

    def get_fix_internal_constraint(self, bbs_used):
        blocks_in_order = [self.building_blocks[index] for index in bbs_used]
        index_offsets = np.cumsum([0] + [len(bb) for bb in blocks_in_order])

        all_bonds = []
        all_angles = []
        all_dihedrals = []

        for bb_index, atom_index_offset in zip(bbs_used, index_offsets):

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

        return FixInternals(bonds=all_bonds, angles=all_angles, dihedrals=all_dihedrals)

    def get_building_block(self, placed, random_rotation=True):
        # Get a building block:
        remaining = np.argwhere((self.N_blocks - placed) > 0).flatten()
        index = int(np.random.choice(remaining, size=1)[0])
        bb = self.building_blocks[index].copy()

        if random_rotation:
            phi0, phi1, phi2 = np.random.uniform(0, 360, size=3)
            bb.rotate(phi2, (0,0,1))
            bb.rotate(phi0, (1,0,0))
            bb.rotate(phi1, (0,1,0))
            
        return bb, index    




