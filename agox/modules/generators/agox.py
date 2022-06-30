import numpy as np
from agox.modules.generators.ABC_generator import GeneratorBaseClass
from agox.modules.databases import Database
from agox.main import AGOX
from agox.observer import ObserverHandler

class AGOXGenerator(GeneratorBaseClass):

    name = 'AGOX generator'

    def __init__(self, modules=[], database=None, main_database=None, iterations=50, **kwargs):
        super().__init__(**kwargs)
        self.modules = modules
        self.iterations = iterations
        self.database = database
        self.main_database = main_database

    def get_candidates(self, sample, environment):
        self.database.reset()
        # Add the data from the main database, such that it can be used for sampling/training. 
        [self.database.store_candidate(candidate, dispatch=False) for candidate in self.main_database.get_all_candidates()]

        for module in self.modules:
            if issubclass(module.__class__, ObserverHandler):
                #module.reset_observers()
                print(module.print_observers())

        self.agox = AGOX(*self.modules)

        print('#'*79)
        print('AGOX GENERATOR STARTING')
        print('#'*79)
        self.agox.run(N_iterations=self.iterations, verbose=True)
        print('#'*79)
        print('AGOX GENERATOR FINISHED')
        print('#'*79)

        # Return only the candidates that we have produced
        candidates = self.database.get_all_candidates()[len(self.main_database):]
        return candidates

    @classmethod
    def get_gofee_generator(cls, environment, database, calculator, iterations=25, 
        prefix='INNER AGOX',  c1=0.7, c2=1.3):
        from agox.modules.generators import RandomGenerator, PermutationGenerator, RattleGenerator
        from agox.modules.samplers import KMeansSampler
        from agox.modules.databases.memory import MemoryDatabase
        from agox.modules.collectors import StandardCollector
        from agox.modules.acquisitors import LowerConfidenceBoundAcquisitor
        from agox.modules.postprocessors import MPIRelaxPostprocess
        from agox.modules.models import ModelGPR
        from agox.modules.postprocessors import WrapperPostprocess
        from agox.modules.evaluators import LocalOptimizationEvaluator

        verbose=False

        model_database = MemoryDatabase(order=6, prefix=prefix, verbose=verbose)

        model_model = ModelGPR.default(environment, model_database)
        model_model.iteration_start_training = 1

        model_sampler = KMeansSampler(feature_calculator=model_model.get_feature_calculator(), 
            database=model_database, order=1, verbose=verbose)

        model_random_generator = RandomGenerator(**environment.get_confinement(), 
            c1=c1, c2=c2, may_nucleate_at_several_places=True, prefix=prefix, verbose=verbose)

        model_rattle_generator = RattleGenerator(**environment.get_confinement(), 
            c1=c1, c2=c2, prefix=prefix, verbose=verbose)

        model_permutation_generator = PermutationGenerator(**environment.get_confinement(), 
            c1=c1, c2=c2, prefix=prefix, verbose=verbose)

        model_collector = StandardCollector(generators=[model_random_generator, model_permutation_generator, model_rattle_generator], 
            sampler=model_sampler, environment=environment, num_candidates={0:[5, 8, 15]}, order=2, verbose=verbose)

        model_acquisitor = LowerConfidenceBoundAcquisitor(model_model, kappa=2, order=4, verbose=verbose)

        model_relaxer = MPIRelaxPostprocess(model_acquisitor.get_acquisition_calculator(model_database), 
            model_database, order=3, optimizer_run_kwargs={'fmax':0.1, 'steps':100}, 
            constraints=environment.get_constraints(), verbose=verbose, start_relax=2)

        wrapper = WrapperPostprocess(order=3.5)

        model_evaluator = LocalOptimizationEvaluator(calculator,
            optimizer_kwargs={'logfile':None}, verbose=True, store_trajectory=True,
            optimizer_run_kwargs={'fmax':0.05, 'steps':1}, gets={'get_key':'prioritized_candidates'}, 
            constraints=environment.get_constraints(), order=5,  prefix=prefix, number_to_evaluate=3)

        model_agox_modules = [model_database, model_collector, model_relaxer, 
            model_evaluator, model_sampler, model_acquisitor, wrapper]

        return cls(modules=model_agox_modules, database=model_database, 
            main_database=database, iterations=iterations)

