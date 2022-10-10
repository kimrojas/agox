from random import Random
import numpy as np
from agox.generators.ABC_generator import GeneratorBaseClass
from agox.databases import Database
from agox.main import AGOX
from agox.observer import ObserverHandler
from agox.evaluators.ABC_evaluator import EvaluatorBaseClass
from ase.calculators.singlepoint import SinglePointCalculator

class AGOXGenerator(GeneratorBaseClass):

    name = 'AGOX generator'

    def __init__(self, modules=[], database=None, main_database=None, iterations=50, recalc=True, **kwargs):
        super().__init__(**kwargs)
        self.modules = modules
        self.iterations = iterations
        self.database = database
        self.main_database = main_database
        self.recalc = recalc
        if self.recalc:
            # search for evaluator among modules
            evaluator = None
            for module in self.modules:
                if issubclass(module.__class__,EvaluatorBaseClass):
                    evaluator = module
                    break
            assert evaluator is not None, 'Cannot find the evaluator for the AGOX recalc generator'

            # get the calculator from the evaluator module
            self.calculator = evaluator.calculator
            
        self.first_call = True

    def _copy_candidates_from_main_database_to_database(self):
        if self.recalc:
            for i,outer_candidate in enumerate(self.main_database.get_all_candidates(respect_worker_number=True)):
                inner_candidate = outer_candidate.copy()
                inner_candidate.set_calculator(self.calculator)
                E = inner_candidate.get_potential_energy()
                single_point_calc = SinglePointCalculator(inner_candidate, energy=E)
                inner_candidate.set_calculator(single_point_calc)
                self.database.store_candidate(inner_candidate, dispatch=False)
        else:
            [self.database.store_candidate(candidate, dispatch=False) for candidate
             in self.main_database.get_all_candidates(respect_worker_number=True)]

    def get_candidates(self, sample, environment):
        self.database.reset()

        # Add the data from the main database, such that it can be used for sampling/training/etc.
        self._copy_candidates_from_main_database_to_database()
        
        self.database.set_number_of_preset_candidates(len(self.database))

        print('#'*79); print('STARTING AGOX GENERATOR'); print('#'*79)
        print(f'LENGTH OF INTERNAL DATABASE: {len(self.database)}')

        self.agox = AGOX(*self.modules)
        self.agox.run(N_iterations=self.iterations, verbose=True, hide_log=True)

        print('#'*79); print('FINISHED AGOX GENERATOR'); print('#'*79)

        # Return only the candidates that we have produced
        candidates = self.database.get_all_candidates(return_preset=False)
        self.first_call = False
        return candidates

    @classmethod
    def get_gofee_generator(cls, environment, database, calculator, iterations=25, 
        number_of_candidates=[1, 0, 0, 0], prefix='INNER AGOX ',  c1=0.7, c2=1.3, model_kwargs={}, 
        additional_modules=None, fix_template=True, constraints=None, generators=None):
        from agox.generators import RandomGenerator, PermutationGenerator, RattleGenerator, SamplingGenerator
        from agox.samplers import KMeansSampler
        from agox.databases.memory import MemoryDatabase
        from agox.collectors import ParallelCollector
        from agox.acquisitors import LowerConfidenceBoundAcquisitor
        from agox.postprocessors import ParallelRelaxPostprocess
        from agox.models import ModelGPR
        from agox.postprocessors import WrapperPostprocess
        from agox.evaluators import LocalOptimizationEvaluator

        if constraints is None:
            constraints = environment.get_constraints()

        verbose = True

        internal_database = MemoryDatabase(order=6, prefix=prefix, verbose=True)

        internal_model = ModelGPR.default(environment, internal_database, **model_kwargs)
        internal_model.iteration_start_training = 1

        internal_sampler = KMeansSampler(feature_calculator=internal_model.get_feature_calculator(), 
            database=internal_database, use_saved_features=True, order=1, verbose=verbose)
        internal_sampler.attach(internal_database)

        ######################################################################################################
        # Generators & Collector
        ######################################################################################################

        if generators is None:

            internal_random_generator = RandomGenerator(**environment.get_confinement(), 
                c1=c1, c2=c2, may_nucleate_at_several_places=True, prefix=prefix, verbose=verbose)

            internal_rattle_generator = RattleGenerator(**environment.get_confinement(), 
                c1=c1, c2=c2, prefix=prefix, verbose=verbose)

            internal_permutation_generator = PermutationGenerator(**environment.get_confinement(), 
                c1=c1, c2=c2, prefix=prefix, verbose=verbose)

            internal_sampling_generator = SamplingGenerator(**environment.get_confinement(), 
                c1=c2, c2=c2, prefix=prefix, verbose=verbose)

            generators = [internal_random_generator, internal_permutation_generator, internal_rattle_generator, 
                internal_sampling_generator]

        internal_collector = ParallelCollector(generators=generators, 
            sampler=internal_sampler, environment=environment, 
            num_candidates={0:number_of_candidates}, order=2, verbose=verbose)

        ######################################################################################################
        # Acquisitor - Evaluator
        ######################################################################################################

        internal_acquisitor = LowerConfidenceBoundAcquisitor(internal_model, kappa=2, order=4, verbose=verbose)

        internal_relaxer = ParallelRelaxPostprocess(internal_acquisitor.get_acquisition_calculator(), 
            order=3, optimizer_run_kwargs={'fmax':0.1, 'steps':100}, 
            verbose=verbose, start_relax=2, fix_template=fix_template, constraints=constraints)

        internal_wrapper = WrapperPostprocess(order=3.5)

        internal_evaluator = LocalOptimizationEvaluator(calculator,
            optimizer_kwargs={'logfile':None}, verbose=True, store_trajectory=True,
            optimizer_run_kwargs={'fmax':0.05, 'steps':1}, gets={'get_key':'prioritized_candidates'}, 
            order=5,  prefix=prefix, number_to_evaluate=3, 
            fix_template=fix_template, constraints=constraints)

        ######################################################################################################
        # Modules:
        ######################################################################################################

        internal_agox_modules = [internal_database, internal_collector, internal_relaxer, 
            internal_evaluator, internal_acquisitor, internal_wrapper]

        if additional_modules is not None:
            internal_agox_modules += additional_modules

        return cls(modules=internal_agox_modules, database=internal_database, 
            main_database=database, iterations=iterations)

    @classmethod
    def get_rss_generator(cls, environment, database, calculator, iterations=25, prefix='INNER AGOX ', 
        additional_modules=None, c1=0.7, c2=1.3, model_kwargs={}):
        from agox.generators import RandomGenerator
        from agox.databases.memory import MemoryDatabase
        from agox.collectors import StandardCollector
        from agox.postprocessors import RelaxPostprocess
        from agox.models import ModelGPR
        from agox.postprocessors import WrapperPostprocess
        from agox.evaluators import LocalOptimizationEvaluator

        verbose=True

        internal_database = MemoryDatabase(order=6, prefix=prefix, verbose=True)

        internal_generator = RandomGenerator(**environment.get_confinement(), 
            c1=c1, c2=c2, may_nucleate_at_several_places=True, prefix=prefix, verbose=verbose)

        internal_collector = StandardCollector(generators=[internal_generator], 
            sampler=None, environment=environment, num_candidates={0:[1]}, order=2, verbose=verbose)

        internal_model = ModelGPR.default(environment, internal_database, **model_kwargs)
        internal_model.iteration_start_training = 1

        internal_relaxer = RelaxPostprocess(model=internal_model, order=3)

        internal_wrapper = WrapperPostprocess(order=3.5)

        internal_evaluator = LocalOptimizationEvaluator(calculator,
            optimizer_kwargs={'logfile':None}, verbose=True, store_trajectory=True,
            optimizer_run_kwargs={'fmax':0.05, 'steps':5}, gets={'get_key':'candidates'}, 
            constraints=environment.get_constraints(), order=5,  prefix=prefix, number_to_evaluate=1)

        internal_agox_modules = [internal_database, internal_collector, internal_relaxer, 
            internal_evaluator, internal_wrapper]

        if additional_modules is not None:
            internal_agox_modules += additional_modules

        return cls(modules=internal_agox_modules, database=internal_database, 
            main_database=database, iterations=iterations)

    def add_module(self, module):
        self.modules.append(module)

    
