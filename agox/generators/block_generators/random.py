from agox.generators.block_generators.ABC_block import BlockGeneratorBaseClass
import numpy as np

class RandomBlockGenerator(BlockGeneratorBaseClass):

    name = 'BlockRandomGenerator'
                        
    def get_candidates(self, sampler, environment):
        template = environment.get_template() 
        candidate = template.copy()

        placed = np.zeros(len(self.building_blocks))
        block_indices = []
        blocks_used = []
        while np.sum(placed) < np.sum(self.N_blocks):
            block, block_index = self.get_building_block(placed)
            
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
            
            block.positions += suggested_position

            # Checks confinement for all atoms in the building block at once.
            if not self.check_confinement(block.positions):
                continue
            
            # Checks that distances are not too bad.
            # This implementation does NOT care about PBCs.
            if not self.check_distances(candidate, block, present=False):
                continue
            
            block_indices += [(np.arange(len(candidate), len(candidate)+len(block)))]
            candidate += block
            placed[block_index] += 1
            blocks_used.append(block_index)

        candidate = self.convert_to_candidate_object(candidate, template)
        candidate.add_meta_information('blocks_used', np.array(blocks_used))
        candidate.add_meta_information('block_indices', np.array(block_indices, dtype=object))

        if self.apply_constraint:
            constraint = self.get_fix_internal_constraint(blocks_used)
            candidate.set_constraint(constraint)

            return [candidate]

        return [candidate]

    def get_building_block(self, placed):
        # Get a building block:
        remaining = np.argwhere((self.N_blocks - placed) > 0).flatten()
        index = int(np.random.choice(remaining, size=1)[0])
        bb = self.building_blocks[index].copy()

        phi0, phi1, phi2 = np.random.uniform(0, 360, size=3)
        bb.rotate(phi2, (0,0,1))
        bb.rotate(phi0, (1,0,0))
        bb.rotate(phi1, (0,1,0))
        return bb, index