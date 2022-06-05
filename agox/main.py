# Copyright 2021-2022, Mads-Peter V. Christiansen, Bjørk Hammer, Nikolaj Rønne. 
# This file is part of AGOX.
# AGOX is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# AGOX is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should 
# have received a copy of the GNU General Public License along with AGOX. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pickle
from agox.modules.candidates.standard import StandardCandidate
from agox.observer import ObserverHandler, FinalizationHandler
from agox.modules.helpers.helper_observers.logger import Logger

from agox.modules.helpers.writer import header_footer, Writer, ICON, header_print

VERSION = "1.1.0"

class AGOX(ObserverHandler, FinalizationHandler, Writer):
    """
    AGO-X
    Atomistic Global Optimization X
    """
    def __init__(self, *args, **kwargs):
        """
        Observers are supplied through *args.

        Supported **kwargs:
            - seed: Random seed for numpy.
            - use_log: Boolean - use logger or not.
        """
        ObserverHandler.__init__(self)
        FinalizationHandler.__init__(self)
        Writer.__init__(self, verbose=True, use_counter=False, prefix='')

        print(ICON.format(VERSION))
        header_print('Initialization starting')

        seed = kwargs.pop('seed', None)
        if seed is not None:
            np.random.seed(seed)
            self.writer('Numpy random seed: {}'.format(seed))

        self.elements = args
        self.candidate_instantiator = StandardCandidate
        self._iteration_counter = 0

        self.check_elements()
        self._update()

        use_log = kwargs.pop('use_log', None)
        if use_log is not False:
            logger = Logger()
            logger.attach(self)
            logger.assign_from_main(self)
        
        unused_keys = False
        for key, value in kwargs.items():
            self.writer("Unused kwarg '{}' given with value {}".format(key, value))
            unused_keys = True
        if unused_keys:
            self.writer('Stopping due to unused keys as behavior may not be as expected')
            exit()

    def set_candidate_instantiator(self, candidate_instantiator):
        self.candidate_instantiator = candidate_instantiator

    @property
    def iteration_counter(self):
        return self._iteration_counter
    
    @iteration_counter.setter
    def iteration_counter(self, value):
        if isinstance(value, int):
            self._iteration_counter = value
        else:
            raise TypeError('iteration counter must be an integer')

    def get_iteration_counter(self):
        return self.iteration_counter
 
    def _update(self):
        for element in self.elements:
            if hasattr(element, 'attach'):
                element.attach(self)
            if hasattr(element, 'assign_from_main'):
                element.assign_from_main(self)
    
    def check_elements(self):
        checked_elements = []
        for element in self.elements:
            if type(element) == list:
                if len(element) > 1:
                    combined_element = element[0] + element[1]
                    checked_elements.append(combined_element)
                else:
                    checked_elements.append(element[0])
            else:
                checked_elements.append(element)
        self.elements = checked_elements

    def run(self, N_iterations, verbose=True, hide_log=True):
        """
        Function called by runscripts that starts the actual optimization procedure. 

        This function is controlled by modules attaching themselves as observers to this module. 
        The order system ensures that modules with lower order are executed first, but no gurantee within each order, 
        so if two modules attach themselves both with order 0 then their individual execution order is not guranteed. 
        However, an observer with order 0 will always be executed before an observer with order 1. 

        The default ordering system is: 
        order = 0: Execution order

        All modules that intend to attach themselves as observers MUST take the order as an argument (with a default value(s)), 
        so that if a different order is wanted that can be controlled from runscripts. Do NOT change order default values!
        """

        if verbose: 
            self.print_observers(hide_log=hide_log)
            self.observer_reports(hide_log=hide_log)
            header_print('Initialization finished')
        # Main-loop calling the relevant observers.   
        converged = False
        while self.iteration_counter < N_iterations and not converged: 
            self.iteration_counter += 1
            self.iteration_cache = {}
            print('\n\n')
            self.header_print('Iteration: {}'.format(self.iteration_counter))
            for observer in self.get_observers_in_execution_order():
                state = observer()
                if state is not None: 
                    if state is True:
                        converged = True
            self.header_print('Iteration finished')
        # Some things may want to perform some operation only at the end of the run. 
        for method in self.get_finalization_methods():
            method()

    ####################################################################################################################
    # iteration Cache Methods:
    ####################################################################################################################

    def get_from_cache(self, key):
        return self.iteration_cache.get(key)

    def add_to_cache(self, key, data, mode):
        """
        modes: 
            w: Will overwrite existing data with the same key. 
            a: Will append to existing data (if there is existing data). 
        """        
        assert(type(data) == list)

        if key in self.iteration_cache.keys() and mode != 'w':
            self.iteration_cache[key] += data
        else:
            self.iteration_cache[key] = data
        
