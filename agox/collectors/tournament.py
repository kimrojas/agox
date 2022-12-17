from agox.collectors.standard import StandardCollector
from timeit import default_timer as dt
from agox.observer import Observer
from agox.writer import Writer, agox_writer

class TournamentCollector(StandardCollector):

    def __init__(self, gets={'get_nonsamplable': 'nonsamplable_inner_candidates',
                             'get_samplable': 'samplable_inner_candidates'}, **kwargs):
        super().__init__(gets=gets, **kwargs)

    name = 'TournamentCollector'

    @agox_writer
    @Observer.observer_method
    def generate_candidates(self, state):
        nonsamplable_candidates = state.get_from_cache(self, self.get_nonsamplable) or []
        self.writer('LEN nonsamplable_candidates: ',len(nonsamplable_candidates))
        samplable_candidates = state.get_from_cache(self, self.get_samplable) or []
        self.writer('LEN samplable_candidates: ',len(samplable_candidates))
    
        if self.do_check():
            candidates = self.make_candidates(nonsamplable_candidates, samplable_candidates)

        # Add to the iteration_cache:
        state.add_to_cache(self, self.set_key, candidates, 'a')
        self.writer('Number of candidates this iteration: {}'.format(len(candidates)))

    def make_candidates(self, nonsamplable_candidates, samplable_candidates):

        all_candidates = []
        for generator, num_candidates in zip(self.generators, self.get_number_of_candidates()):
            print('GENERATOR',generator)
            for sample in range(num_candidates):
                candidates = generator(sampler=self.sampler, environment=self.environment, nonsamplable_candidates=nonsamplable_candidates, samplable_candidates=samplable_candidates)
                    
                for candidate in candidates:
                    all_candidates.append(candidate)
        
        # The generator may have returned None if it was unable to build a candidate.
        all_candidates = list(filter(None, all_candidates))
                
        return all_candidates
