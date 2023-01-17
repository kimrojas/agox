import numpy as np
import pytest 
from agox.environments import Environment
from agox.samplers import MetropolisSampler
from ase import Atoms
from agox.generators import RandomGenerator, RattleGenerator, CenterOfGeometryGenerator, ReplaceGenerator, ReuseGenerator, SamplingGenerator, PermutationGenerator
from agox.candidates import CandidateBaseClass

from environment_test import environment_and_dataset

@pytest.fixture
def simple_settings():
    return {'cell':np.eye(3) * 10,  'corner':np.zeros(3)}

@pytest.fixture
def simple_environment(simple_settings):
    cell = simple_settings['cell']
    corner = simple_settings['corner']
    template = Atoms(cell=cell)
    return Environment(template, symbols='H2C2', confinement_cell=cell, confinement_corner=corner)

@pytest.fixture
def simple_candidate(simple_environment, simple_settings):
    return RandomGenerator(**simple_environment.get_confinement())(None, simple_environment)[0]

@pytest.fixture
def simple_sampler(simple_candidate):
    sampler = MetropolisSampler()
    sampler.sample = [simple_candidate]
    return sampler

@pytest.mark.parametrize('generator_class', [RandomGenerator])#, RattleGenerator, ReplaceGenerator, CenterOfGeometryGenerator])
class TestGenerator:

    def assertions(self, candidates, environment, sampler):
        for candidate in candidates:
            assert issubclass(candidate.__class__, CandidateBaseClass)
            assert len(candidate) == len(environment.get_all_numbers())
            assert (candidate.cell == environment.get_template().get_cell()).all()

    def setup_generator(self, generator_class, environment, **kwargs):
        return generator_class(**environment.get_confinement(), **kwargs)

    def setup_sampler(self, dataset):
        sampler = MetropolisSampler()
        sampler.sample = [dataset[0]]
        return sampler

    def test_generators(self, generator_class, environment_and_dataset):
        environment, dataset = environment_and_dataset
        
        generator = self.setup_generator(generator_class, environment)
        sampler = self.setup_sampler(dataset)
        candidates = [None]
        for i in range(1):
            candidates = generator(sampler, environment)
            if not candidates[0] == None:
                break
        self.assertions(candidates, environment, sampler)



    # def test_random_generator(self, simple_environment, simple_sampler):
    #     generator = self.setup_generator(RandomGenerator, simple_environment)
    #     candidates = generator(simple_sampler, simple_environment)
    #     self.assertions(candidates, simple_environment, simple_sampler)

    # def test_rattle_generator(self, simple_sampler, simple_environment):
    #     generator = self.setup_generator(RattleGenerator, simple_environment)
    #     candidates = generator(simple_sampler, simple_environment)

    #     self.assertions(candidates, simple_environment, simple_sampler)

    # def test_replace_generator(self, simple_sampler, simple_environment):
    #     generator = self.setup_generator(ReplaceGenerator, simple_environment)
    #     candidates = generator(simple_sampler, simple_environment)

    #     self.assertions(candidates, simple_environment, simple_sampler)

    # def test_cog_generator(self, simple_sampler, simple_environment):
    #     generator = self.setup_generator(CenterOfGeometryGenerator, simple_environment)
    #     candidates = generator(simple_sampler, simple_environment)

    #     self.assertions(candidates, simple_environment, simple_sampler)

    # def test_reuse_generator(self, simple_sampler, simple_environment):
    #     generator = self.setup_generator(ReuseGenerator, simple_environment)
    #     candidates = generator(simple_sampler, simple_environment)

    #     self.assertions(candidates, simple_environment, simple_sampler)

