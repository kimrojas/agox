from agox.generators.ABC_generator import GeneratorBaseClass
from ase.data import covalent_radii
from ase import Atom
import numpy as np

from scipy.spatial.distance import cdist

class BuildingBlockGenerator(GeneratorBaseClass):

    name = 'BuildingBlockGenerator'

    def __init__(self, building_blocks, N, **kwargs):
        super().__init__(**kwargs)
        self.building_blocks = building_blocks
        self.N = np.array(N)

        for building_block in self.building_blocks:
            building_block.positions -= building_block.get_center_of_mass()

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
        while np.sum(placed) < np.sum(self.N):
            print(count); count += 1

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

            for pos, num in zip(bb.positions, bb.numbers):
                if not self.check_confinement(pos):
                    build_succesful = False
                    continue
                
                if np.sum(placed) > 0:
                    if not self.check_new_position(candidate, pos, num):
                       build_succesful = False

            if build_succesful:
                bb_count = np.sum(placed)
                bb_indices += [bb_count] * len(bb)
                candidate += bb
                placed[bb_index] += 1
            
        candidate = self.convert_to_candidate_object(candidate, template)
        candidate.add_meta_information('description', self.name)

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

    #def prepare(self):




