from agox.modules.collectors.collector_ABC import CollectorBaseClass
from ase.io import write

from timeit import default_timer as dt


class StandardCollector(CollectorBaseClass):

    name = 'StandardCollector'

    def __init__(self, generators, sampler, environment, num_samples, verbose=False, report_timing=False, **kwargs):
        super().__init__(**kwargs)
        self.generators = generators
        self.sampler = sampler
        self.environment = environment
        self.num_samples = num_samples
        self.verbose = verbose
        self.report_timing = report_timing

    def make_candidates(self):
        self.make_candidate_collection()

    def num_samples_this_episode(self):
        episode = self.get_episode_counter()
        print('NUM_SAMPLES IN EPISODE',episode,':', self.num_samples)
        return self.num_samples

    def make_candidate_collection(self):

        all_candidates = []

        t0 = dt()
        for generator, num_samples in zip(self.generators, self.num_samples_this_episode()):

            for sample in range(num_samples):
                candidates = generator(sampler=self.sampler, environment=self.environment)
                    
                for candidate in candidates:
                    all_candidates.append(candidate)
        
        # The generator may have returned None if it was unable to build a candidate.
        all_candidates = list(filter(None, all_candidates)) 

        if self.report_timing:
            print('Candidate generation time: {}'.format(dt()-t0))

        if self.verbose:
            write('candidate_ensemple_raw_{:05d}.traj'.format(self.get_episode_counter()),all_candidates)

        self.candidates = all_candidates
        
        print('Number of candidates this episode: {}'.format(len(self.candidates)))
     
    def assign_from_main(self, main):
        super().assign_from_main(main)
        for generator in self.generators:
            generator.assign_from_main(main)
        

