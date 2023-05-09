"""
A database records the candidates that have been evaluated during a search. 

Further, because some things need to be trained, e.g. a ML potential model, 
databases can be used to store the training data. The databases all implement
a similar observer-pattern as the main AGOX class, so models can be trained 
whenever data is added to the database.
"""

from .ABC_database import DatabaseBaseClass
from agox.databases.database import Database 
from agox.databases.database_concurrent import ConcurrentDatabase
