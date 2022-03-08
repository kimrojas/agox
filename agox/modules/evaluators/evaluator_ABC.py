from abc import ABC, abstractmethod
from agox.observer_handler import Observer

class EvaluatorBaseClass(ABC, Observer):

    def __init__(self, number_to_evaluate=1, gets={'get_key':'prioritized_candidates'}, sets={'set_key':'evaluated_candidates'}, order=5):
        super().__init__(gets=gets, sets=sets, order=order)
        self.number_to_evaluate = number_to_evaluate
    
    def __call__(self, candidate):
        return self.evaluate_candidate(candidate)
        
    @abstractmethod
    def evaluate_candidate(self, candidate):
        """
        Performs a 'gauge'-operation on the candidate object IN PLACE and returns a boolean state that describes 
        whether the candidate parsed the gauge successfully. 
        """
        return state

    @property
    @abstractmethod
    def name(self):
        pass

    def evaluate_candidates(self):

        candidates = self.get_from_cache(self.get_key)
        done = False

        self.evaluated_candidates = []
        while candidates and not done:            

            candidate = candidates.pop(0)            
            state = self.evaluate_candidate(candidate)

            if state:
                self.evaluated_candidates.append(candidate)

                if len(self.evaluated_candidates) == self.number_to_evaluate:
                    done = True

        self.add_to_cache(self.set_key, self.evaluated_candidates, mode='a')

    def __add__(self, other):
        from .evaluator_collection import GaugeCollection
        return GaugeCollection(gauges=[self, other])
                
    def attach(self, main):
        main.attach_observer(self.name+'.evaluate_candidates', self.evaluate_candidates, order=self.order)

    def assign_from_main(self, main):
        super().assign_from_main(main)



