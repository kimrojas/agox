from .ABC_evaluator import EvaluatorBaseClass
from agox.evaluators.local_optimization import LocalOptimizationEvaluator
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator

class SinglePointEvaluator(LocalOptimizationEvaluator):

    name = 'SinglePointEvaluator'

    def __init__(self, calculator, **kwargs): 
        super().__init__(calculator, optimizer_run_kwargs=dict(steps=0), **kwargs)