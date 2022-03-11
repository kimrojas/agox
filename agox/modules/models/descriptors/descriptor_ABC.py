from abc import ABC, abstractmethod

class DescriptorBaseClass(ABC):

    def __init__(self):
        pass

    def __call__(self, atoms):
        return self.get_feature(atoms)
    
    @abstractmethod
    def get_feature(self, atoms):
        return feature

    def get_dimension(self, atoms):
        if hasattr(self, 'ndim'):
            return self.ndim
        else:
            print('Dimension not available')
            return

    
