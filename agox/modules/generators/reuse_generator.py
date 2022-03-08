from agox.modules.generators.generator_ABC import GeneratorBaseClass
from ase.data import covalent_radii
from ase import Atom
import numpy as np

class ReuseGenerator(GeneratorBaseClass):

    name = 'ReuseGenerator'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_candidates(self, sample, environment):
        candidate = self.collector.pop_candidate()

        if candidate is None:
            return [None]

        description = self.name + candidate.get_meta_information('description')
        candidate.add_meta_information('description', description)

        return [candidate]

    def assign_from_main(self, main):
        super().assign_from_main(main)
        self.collector = main.collector