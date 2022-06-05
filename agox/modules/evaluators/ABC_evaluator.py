from abc import ABC, abstractmethod
from agox.observer import Observer
from agox.modules.helpers.writer import header_footer, Writer


class EvaluatorBaseClass(ABC, Observer, Writer):

    def __init__(self, number_to_evaluate=1, gets={'get_key':'prioritized_candidates'}, sets={'set_key':'evaluated_candidates'}, 
        order=5, verbose=True, use_counter=True, prefix=''):
        Observer.__init__(self, gets=gets, sets=sets, order=order)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)
        self.number_to_evaluate = number_to_evaluate

        self.add_observer_method(self.evaluate_candidates, self.sets[0], self.gets[0], self.order[0])
    
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

    @header_footer
    def evaluate_candidates(self):

        candidates = self.get_from_cache(self.get_key)
        done = False

        self.evaluated_candidates = []
        passed_evaluation_count = 0
        while candidates and not done:            

            candidate = candidates.pop(0)            
            state = self.evaluate_candidate(candidate)

            if state:
                self.evaluated_candidates_append(candidate)
                passed_evaluation_count += 1

                if passed_evaluation_count == self.number_to_evaluate:
                    done = True

        self.add_to_cache(self.set_key, self.evaluated_candidates, mode='a')

    def evaluated_candidates_append(self, candidate):
        self.evaluated_candidates.append(candidate)

    def __add__(self, other):
        return EvaluatorCollection(evaluators=[self, other])

    def assign_from_main(self, main):
        super().assign_from_main(main)

class EvaluatorCollection(EvaluatorBaseClass): 

    name = 'EvaluatorCollection'

    def __init__(self, evaluators):
        super().__init__()
        self.evaluators = evaluators

    def evaluate_candidate(self, candidate):
        state = self.apply_evaluators(candidate)
        return state

    def add_evaluator(self, evaluator):
        self.evaluators.append(evaluator)

    def list_evaluators(self):
        for i, evaluator in enumerate(self.evaluators):
            print('Evaluator {}: {} - {}'.format(i, evaluator.name, evaluator))

    def apply_evaluators(self, candidate):
        for evaluator in self.evaluators:
            evaluator_state = evaluator(candidate)
            if not evaluator_state:
                return False
        return True
    
    def assign_from_main(self, main):
        super().assign_from_main(main)
        for evaluator in self.evaluators:
            evaluator.assign_from_main(main)

    def __add__(self, other):
        self.evaluators.append(other)
        return self
