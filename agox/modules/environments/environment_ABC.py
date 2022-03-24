from abc import ABC, abstractmethod

class EnvironmentBaseClass(ABC):
    """
    The Environment contains important properties about the envrionment (or conditions) of the global atomisation problem. 
    These are at least: 

    - numbers: The atomic numbers of missings atoms e.g. C2H4 is [1, 1, 1, 1, 6, 6]. 
    - template: An ASE Atoms object that describes the *static* atoms, e.g a surface or part of a molecule.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_template(self, **kwargs):
        pass

    @abstractmethod    
    def get_numbers(self, numbers, **kwargs):
        pass
    
    def assign_from_main(self, main):
        pass

    def attach(self, main):
        pass