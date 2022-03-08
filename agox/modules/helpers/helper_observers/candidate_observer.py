import numpy as np
import pickle

class CandidateObserver:

    name = 'CandidateObserver'

    def __init__(self, collector, database, record_fitness=True, record_description=True):
        self.collector = collector
        self.database = database
        self.fitness = []
        self.descriptions = []

        self.record_fitness = record_fitness
        self.record_description = record_description

    def record_candidate_statistics(self):
        """
        Record the statistics of generated candidates.
        - Fitness                                              (number, N x 1)
        - Generator type.                                      (String, N x 1)
        - Was it selected?                                     (Boolean, N x 1)

        With N candidates in an episode.

        Not yet sure how to record energies and uncertainties in a reasonable fashion.

        Assumes only 1 true energy calculation is done pr. episode. 

        Not efficient.
        """

        # Get candidates from candidate ensemble:
        candidates = self.collector.get_current_candidates()
        selected_structure = self.database.get_most_recent_candidate() 

        # Get descriptions:
        if self.record_description:
            descriptions = [candidate.get_meta_information('description') for candidate in candidates]
            descriptions = [selected_structure.get_meta_information('description')] + descriptions
            self.descriptions.append(descriptions)
            # The first structure is always the one selected by the acquisition function.

        if self.record_fitness:
            try:
                fitness = self.collector.values
                self.fitness.append(fitness)
            except:
                pass                    
    
    def easy_attach(self, agox, order=3):
        agox.attach_observer(self.name,self.record_candidate_statistics, order=order)

    def save_recordings(self):
        if self.record_description:
            with open('descriptions.pckl', 'wb') as f:
                pickle.dump(self.descriptions, f)


        print(self.fitness)
        if self.record_fitness:
            with open('fitness.pckl', 'wb') as f:
                pickle.dump(self.fitness, f)
            




