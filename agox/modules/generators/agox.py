import numpy as np
from agox.modules.generators.ABC_generator import GeneratorBaseClass
from agox.modules.databases import Database

class AGOXGenerator(GeneratorBaseClass):

    name = 'AGOX generator'

    def __init__(self, agox, database, model, iterations=50, **kwargs):
        super().__init__(**kwargs)
        self.agox = agox
        self.iterations = iterations
        self.database = database
        self.model = model

    def get_candidates(self, sample, environment):
        self.agox.iteration_counter = 0
        self.agox.iteration_cache = {}
        candidate = sample.get_random_member()
        candidate.set_calculator(self.model)
        self.database.candidates = [candidate]
        self.database.candidate_energies = [candidate.get_potential_energy()]
        self.agox.run(N_iterations=self.iterations, verbose=True)
        return self.database.candidates[1:]

