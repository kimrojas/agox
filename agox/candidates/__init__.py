"""
A candidate is the basic piece of data being moved around in the algorithm. 
Generators create candidates, and evaluators evaluate the objective function 
at the coordinates described by the candidate. 
"""

from .ABC_candidate import CandidateBaseClass
from .standard import StandardCandidate