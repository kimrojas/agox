import numpy as np
from agox.modules.generators.ABC_generator import GeneratorBaseClass
from agox.modules.databases import Database
from agox.main import AGOX

class AGOXGenerator(GeneratorBaseClass):

    name = 'AGOX generator'

    def __init__(self, params=[], database=None, main_database=None, iterations=50, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.iterations = iterations
        self.database = database
        self.main_database = main_database

    def get_candidates(self, sample, environment):
        self.database.reset()
        # Add the data from the main database, such that it can be used for sampling/training. 
        [self.database.store_candidate(candidate, dispatch=False) for candidate in self.main_database.get_all_candidates()]

        print('##### LENGTH OF DATABASE BEFORE: {}'.format(len(self.database)))

        self.agox = AGOX(*self.params)
        self.agox.iteration_counter = 0
        self.agox.iteration_cache = {}
        self.database.candidate_energies = []
        self.agox.run(N_iterations=self.iterations, verbose=True)

        # Return only the candidates that we have produced
        print('##### LENGTH OF DATABASE AFTER: {}'.format(len(self.database)))
        print('##### LENGTH OF MAIN AFTER: {}'.format(len(self.main_database)))

        return self.database.get_all_candidates()[len(self.main_database):]

