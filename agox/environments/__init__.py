"""
An environment describes the search problem
- How many and which types of atoms will the search place? 
- What is the computational cell? 
- Is there a fixed template, e.g. a surface to be decorated by the search? 
- Is the search constrained to a certain region of space?

These are settings that are not part of the search algorithm itself, but 
rather describe the problem that the search is trying to solve.
"""
from .ABC_environment import EnvironmentBaseClass
from agox.environments.environment import Environment