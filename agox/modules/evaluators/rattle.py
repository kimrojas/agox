from agox.modules.evaluators.single_point import EnergyEvaluator
import numpy as np


class RattleEvaluator(EnergyEvaluator):

    def __init__(self, calculator, seed=0, constraints=[], stdev=0.1, skip_frequency=10, **kwargs): 
        super().__init__(calculator=calculator, **kwargs)
        self.seed = seed
        self.constraints = constraints
        self.stdev = stdev
        self.skip_frequency = skip_frequency


    def evaluate_candidate(self, candidate):
        if self.get_iteration_counter() % self.skip_frequency > 0:
            candidate.set_constraint(self.constraints)

            self.writer('Im rattling with stdev:', self.stdev)
                
            seed = self.seed*10000 + self.get_iteration_counter()
            candidate.rattle(stdev=self.stdev, seed=seed)
            candidate.set_constraint()
            
        return super().evaluate_candidate(candidate)



    def assign_from_main(self, main):
        super().assign_from_main(main)
        self.get_iteration_counter = main.get_iteration_counter
