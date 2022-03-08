import numpy as np
import pickle

from agox.modules.databases import Database
from agox.modules.candidates.candidate_standard import StandardCandidate
from agox.observer_handler import ObserverHandler, FinalizationHandler
from copy import deepcopy

class AGOX(ObserverHandler, FinalizationHandler):
    """
    AGO-X
    Atomistic Global Optimization X
    """     
    def __init__(self, candidate_instantiator=None, seed=None, **kwargs):
        ObserverHandler.__init__(self)
        FinalizationHandler.__init__(self)

        np.random.seed(seed)

        self.elements = kwargs

        self.candidate_instantiator = StandardCandidate if candidate_instantiator is None else candidate_instantiator
        self._episode_counter = 0

        self.check_elements()
        self._update()


    @property
    def episode_counter(self):
        return self._episode_counter
    
    @episode_counter.setter
    def episode_counter(self, value):
        if isinstance(value, int):
            self._episode_counter = value
        else:
            raise TypeError('Episode counter must be an integer')

    def get_episode_counter(self):
        return self.episode_counter
 
    def _update(self):
        for element in self.elements:
            if hasattr(element, 'attach'):
                element.attach(self)
            if hasattr(element, 'assign_from_main'):
                element.assign_from_main(self)
    
    def check_elements(self):
        checked_elements = []
        for name, element in self.elements.items():
            if type(element) == list:
                for j, sub_element in enumerate(element):
                    sub_element.order += 0.1*j
                    checked_elements.append(sub_element)
            else:
                checked_elements.append(element)
        self.elements = checked_elements

    def run(self, N_episodes, verbose=True):
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
            self.print_observers()
            self.observer_reports()

        # Main-loop calling the relevant observers.   
        converged = False
        while self.episode_counter < N_episodes and not converged: 
            self.episode_counter += 1
            self.episode_cache = {}
            print('\n\n Episode: {}'.format(self.episode_counter))
            for observer in self.get_observers_in_execution_order():
                state = observer()
                if state is not None: 
                    if state is True:
                        converged = True

        # Some things may want to perform some operation only at the end of the run. 
        for method in self.get_finalization_methods():
            method()

    def save_seed(self, postfix, model_dir = 'model_checkpoint'):
        f = open(f'{model_dir}/np_state{postfix}', 'wb')
        pickle.dump(np.random.get_state(), f)
        f.close()
    
    def load_seed(self, postfix, model_dir = 'model_checkpoint'):
        f = open(f'{model_dir}/np_state{str(postfix).zfill(6)}', 'rb')
        state = pickle.load(f)
        np.random.set_state(state)
        f.close()

    ####################################################################################################################
    # Episode Cache Methods:
    ####################################################################################################################

    def get_from_cache(self, key):
        return self.episode_cache.get(key)

    def add_to_cache(self, key, data, mode):
        """
        modes: 
            w: Wi <ll overwrite existing data with the same key. 
            a: Will append to existing data (if there is existing data). 
        """        
        assert(type(data) == list)

        if key in self.episode_cache.keys() and mode is not 'w':
            self.episode_cache[key] += data
        else:
            self.episode_cache[key] = data
        
