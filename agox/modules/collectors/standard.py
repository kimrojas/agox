from agox.modules.collectors.ABC_collector import CollectorBaseClass
from timeit import default_timer as dt

class StandardCollector(CollectorBaseClass):

    name = 'StandardCollector'

    def __init__(self, generators, sampler, environment, num_candidates, verbose=True, **kwargs):
        super().__init__(generators, sampler, environment, **kwargs)
        self.num_candidates = num_candidates
        self.verbose = verbose

    def make_candidates(self):
        self.make_candidate_collection()

    def get_number_of_candidates(self):
        if type(self.num_candidates) == list:
            return self.num_candidates
        elif type(self.num_candidates) == dict:
            return self.get_number_of_candidates_for_iteration()
            
    def get_number_of_candidates_for_iteration(self):
        # self.num_candidates must have this form: {0: [], 500: []}
        keys = list(self.num_candidates.keys())
        keys.sort()
        iteration = self.get_iteration_counter()

        num_candidates = self.num_candidates[0] # yes, it must contain 0
        # now step through the sorted list (as long as the iteration is past the key) and extract the num_candidates
        # the last one extracted will be the most recent num_candidates enforced and should apply to this iteration
        for k in keys:
            if iteration < k:
                break
            num_candidates = self.num_candidates[k]
        return num_candidates

    def make_candidate_collection(self):

        all_candidates = []

        t0 = dt()
        for generator, num_candidates in zip(self.generators, self.get_number_of_candidates()):

            for sample in range(num_candidates):
                candidates = generator(sampler=self.sampler, environment=self.environment)
                    
                for candidate in candidates:
                    all_candidates.append(candidate)
        
        # The generator may have returned None if it was unable to build a candidate.
        all_candidates = list(filter(None, all_candidates)) 
        
        self.candidates = all_candidates
        
        self.writer('Number of candidates this iteration: {}'.format(len(self.candidates)))
     
    def assign_from_main(self, main):
        super().assign_from_main(main)
        for generator in self.generators:
            generator.assign_from_main(main)
        

