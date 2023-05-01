from agox.observer import Observer
from agox.writer import Writer, agox_writer
from abc import ABC, abstractmethod

class AttractorMethodBaseClass(ABC, Observer, Writer):
    
    dynamic_attributes = ['structures']

    def __init__(self, order = 5):
        Observer.__init__(self, order = order)
        Writer.__init__(self)
        self.add_observer_method(self.update, order=self.order[0], 
            sets={}, gets={})
        
        self.structures = []
    
    @agox_writer
    @Observer.observer_method
    def update(self, database, state):
        self.structures = database.get_all_candidates()

    @abstractmethod
    def get_attractors(self, database, state):
        pass