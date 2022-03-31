import numpy as np
from agox.modules.acquisitors.acquisitor_ABC import AcquisitorBaseClass

class DummyAcquisitor(AcquisitorBaseClass):

    name = 'DummyAcquisitor'

    """
    This is a BAD acquisitor, but it has some usefulness - e.g. it can be used in conjunction with OnehotSimilarityGauge 
    to approximate the original ASLA behaviour of not accepting a duplicate structure (Although it is less efficient because 
    the current Candidate Ensembles dont dynamically create more candidates but create them all at once)).

    Can also be used for runs where you simply only produce one candidate pr. episode and therefore dont need a acquisition 
    function. 
    """

    def sort_according_to_fitness(self):
        pass


    