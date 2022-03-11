from agox.modules.collectors.collector_ABC import CollectorBaseClass
from ase.io import write
import numpy as np
from timeit import default_timer as dt


class ProbabilityCollector(CollectorBaseClass):

    name = 'ProbabilityCollector'

    def __init__(self, generators, sampler, environment, probabilities, num_candidates=60, include_samples=False, verbose=False, report_timing=False, **kwargs):
        super().__init__(**kwargs)
        self.generators = generators
        self.sampler = sampler
        self.environment = environment
        self.probabilities = probabilities
        self.num_candidates = num_candidates
        self.include_samples = include_samples
        self.verbose = verbose
        self.report_timing = report_timing

    def make_candidates(self):
        self.make_candidate_collection()

    def get_candidate(self, generator):
        for _ in range(5):
            candidate = generator(sampler=self.sampler, environment=self.environment)[0]
            if candidate is not None:
                return candidate
        return None
    
    def make_candidate_collection(self):
        all_candidates = []

        t0 = dt()
        for num in range(self.num_candidates):
            for _ in range(5):
                generator_index = np.random.choice(range(len(self.generators)), p=self.probabilities)
                generator = self.generators[generator_index]
                candidate = self.get_candidate(generator)
                if candidate is not None:
                    all_candidates.append(candidate)
                    break

                
        # The generator may have returned None if it was unable to build a candidate.
        all_candidates = list(filter(None, all_candidates)) 

        if self.report_timing:
            print('Candidate generation time: {}'.format(dt()-t0))

        if self.verbose:
            print(f'Number of candidates generated: {len(all_candidates)}')
            write('candidate_ensemple_raw_{:05d}.traj'.format(self.get_episode_counter()),all_candidates)

        # Leaving this in, but I think this is the wrong way to do this. 
        # If you want to use the sample structures add a generator that does so 
        # It can return None if not wanted every episode! 
        if self.include_samples and self.get_episode_counter()%3:
            all_candidates += self.sampler.sample
        
        self.candidates = all_candidates
        
        if self.verbose:
            write('candidate_ensemple_postproc_{:05d}.traj'.format(self.get_episode_counter()),all_candidates)
    
    def assign_from_main(self, main):
        super().assign_from_main(main)
        self.get_episode_counter = main.get_episode_counter
        for generator in self.generators:
            generator.assign_from_main(main)
        

