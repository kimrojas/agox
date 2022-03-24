from agox.modules.acquisitors.gauges.gauge_ABC import EvaluatorBaseClass

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
 
    def __add__(self, other):
        self.evaluators.append(other)
        return self
