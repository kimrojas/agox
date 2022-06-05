from abc import ABC, abstractmethod
import numpy as np
from ase.io import write
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from agox.observer import Observer

from agox.modules.helpers.writer import header_footer, Writer

class AcquisitorBaseClass(ABC, Observer, Writer):

    """
    Acquisition Base Class. 

    See: Observer-class for information on gets, sets, order. 

    verbose: bool
        Controls how much printing the module does. 

    Abstract methods:

    calculate_acquisition_function: 
        Calculates acquisition function for given candidates.         
    """

    def __init__(self, order=4, gets={'get_key':'candidates'}, sets={'set_key':'prioritized_candidates'}, verbose=True, 
                use_counter=True, prefix=''):   
        Observer.__init__(self, gets=gets, sets=sets, order=order)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)
        self.candidate_list = []

        self.add_observer_method(self.prioritize_candidates, sets=self.sets[0], gets=self.gets[0], order=self.order[0])

    ########################################################################################
    # Required properties
    ########################################################################################    

    @property
    @abstractmethod
    def name(self):
        return NotImplementedError

    ########################################################################################
    # Required methods
    ########################################################################################

    @abstractmethod
    def calculate_acquisition_function(self, candidates):
        """
        Implements the fitness criterion and sorts the candidates stored in self.candidate_list 
        according to this measure. 
        """
        return acquisition_values

    @header_footer    
    def prioritize_candidates(self):
        """
        Method that is attached to the AGOX iteration loop as an observer - not intended for use outside of that loop. 
        """

        # Get data from the iteration data dict. 
        candidate_list = self.get_from_cache(self.get_key)
        candidate_list = list(filter(None, candidate_list))

        # Calculate acquisition function values and sort:
        candidate_list, acquisition_values = self.sort_according_to_acquisition_function(candidate_list)

        # Add the prioritized candidates to the iteration data in append mode!
        self.add_to_cache(self.set_key, candidate_list, mode='a')

        self.print_information(candidate_list, acquisition_values)
            
    ########################################################################################
    # Default methods
    ########################################################################################
        
    def sort_according_to_acquisition_function(self, candidates):        
        """
        Calculates acquisiton-function based on the implemeneted version calculate_acquisition_function. 

        Note: Sorted so that the candidate with the LOWEST acquisition function value is first. 
        """
        acquisition_values = self.calculate_acquisition_function(candidates)
        sort_idx = np.argsort(acquisition_values)
        sorted_candidates = [candidates[i] for i in sort_idx]
        acquisition_values = acquisition_values[sort_idx]
        [candidate.add_meta_information('acquisition_value', acquisition_value) for candidate, acquisition_value in zip(sorted_candidates, acquisition_values)]          
        return sorted_candidates, acquisition_values

    def get_random_candidate(self):
        DeprecationWarning('Will be removed in the future. Please use collector.get_random_candidate')
        return self.collector.get_random_candidate()

    def assign_from_main(self, main):
        super().assign_from_main(main)

    def print_information(self, candidates, acquisition_values):
        pass

    def get_acquisition_calculator(self):
        raise NotImplementedError("'get_acqusition_calculator' is not implemented for this acquisitor")

from ase.calculators.calculator import Calculator, all_changes

class AcquisitonCalculatorBaseClass(Calculator):

    def __init__(self, database, model_calculator, **kwargs):
        super().__init__(**kwargs)
        self.model_calculator = model_calculator
        database.add_objects_to_assign_to(self)
    
    @property
    def verbose(self):
        return self.model_calculator.verbose

    def get_model_parameters(self):
        parameters = self.model_calculator.get_model_parameters()        
        parameters['iteration'] = self.get_iteration_counter()
        return parameters

    def set_model_parameters(self, parameters):
        self.iteration = parameters['iteration']
        self.model_calculator.set_model_parameters(parameters)

    def set_verbosity(self, verbose):
        self.model_calculator.set_verbosity(verbose)

    def get_iteration_number(self):
        if hasattr(self, 'get_iteration_counter'):
            return self.get_iteration_counter()
        else:
            return self.iteration

    def assign_from_main(self, main):
        self.get_iteration_counter = main.get_iteration_counter

    @property
    def ready_state(self):
        return self.model_calculator.ready_state


