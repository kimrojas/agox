from agox.samplers.ABC_sampler import SamplerBaseClass

class SuperSampler(SamplerBaseClass):

    name = 'SuperSampler'

    def __init__(self, sampler=None, meta_key=None, possible_values=None, 
        sampler_kwargs={}, sampler_args=()):

        self.sampler_class = sampler
        self.meta_key = meta_key
        self.possible_values = possible_values

        # We create a sampler for each possible value of meta_key.
        if type(sampler_kwargs) == dict:
            sampler_kwargs_list = [sampler_kwargs for _ in possible_values]
        else:
            sampler_kwargs_list = sampler_kwargs

        if type(sampler_args) == tuple:
            sampler_args_list = [sampler_args for _ in possible_values]
        else:
            sampler_args_list = sampler_args
        
        self.samplers = {}
        for value, args, kwargs in zip(possible_values, sampler_args_list, sampler_kwargs_list):
            self.samplers[value] = self.sampler_class(*args, **kwargs)

    def setup(self, all_candidates):

        # First we split the candidates according to the keys.
        split_candidates = self.split_candidates(all_candidates)

        for key, candidates in split_candidates.items():
            self.samplers[key].setup(candidates)

        self.sample = []
        for sampler in self.samplers.values():
            self.sample += sampler.sample

    def split_candidates(self, all_candidates):

        split_candidates = {val:[] for val in self.possible_values}

        for candidate in all_candidates:
            key = candidate.get_meta_information(self.meta_key)
            assert key in self.possible_values, f'{key} not a possible value for {self.meta_key}'
            split_candidates[key].append(candidate)

        return split_candidates






