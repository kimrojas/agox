"""
Evaluators are responsible for evaluating the objective function of a candidate.
The main two flavours of evaluators are single-point and local optimization.

Single-point evaluators evaluate the objective function at a single point, 
whereas local optimization evaluators perform a local optimization starting from 
the candidate's coordinates.
"""
from .single_point import SinglePointEvaluator
from .local_optimization import LocalOptimizationEvaluator
from .rattle import RattleEvaluator