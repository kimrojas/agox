from agox.generators.ABC_generator import GeneratorBaseClass
from ase.data import covalent_radii
from ase import Atom
import numpy as np

from scipy.spatial.distance import cdist

class BuildingBlockGenerator(GeneratorBaseClass):

    name = 'BuildingBlockGenerator'

    def __init__(self, building_block, N, **kwargs):
        super().__init__(**kwargs)
        self.building_block = building_block
        self.N = N

    def get_candidates(self, sampler, environment):
        template = environment.get_template()
        numbers_list = environment.get_numbers()
        if len(numbers_list) != len(np.repeat(self.building_block.get_atomic_numbers(), self.N)):
            self.writer('Building Blocks and N does not match environment settings!')
            self.writer('Will break now!')
            exit()

        len_of_template = len(template)

        candidate = template.copy()

        placed = 0
        while placed < self.N:
            build_succesful = True
            suggested_position = self.get_box_vector()

            bb = self.get_building_block()
            bb.positions += suggested_position
            for pos, num in zip(bb.positions, bb.numbers):
                if not self.check_confinement(pos):
                    build_succesful = False
                    continue

                if not self.check_new_position(candidate, pos, num):
                    build_succesful = False

            if build_succesful:
                candidate += bb
                placed += 1
            
        
        candidate = self.convert_to_candidate_object(candidate, template)
        candidate.add_meta_information('description', self.name)

        return [candidate]


    def get_building_block(self):
        bb = self.building_block.copy()
        phi0, phi1, phi2 = np.random.uniform(0, 360, size=3)
        bb.rotate(phi2, (0,0,1))
        bb.rotate(phi0, (1,0,0))
        bb.rotate(phi1, (0,1,0))
        return bb
