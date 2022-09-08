from optparse import BadOptionError
import numpy as np
from abc import ABC, abstractmethod

from agox.modules.helpers.writer import header_print, pretty_print

global A
A = 0
def get_next_key():
    """
    Generates a unique always increasing key for observer methods. 

    Returns
    --------
    int: 
        Unique key. 
    """
    global A
    A += 1
    return A

class ObserverHandler:
    """
    Base-class for classes that can have attached observers. 
    """

    def __init__(self):
        self.observers = {}
        self.execution_sort_idx = []

    ####################################################################################################################
    # General handling methods.
    ####################################################################################################################

    def attach_observer(self, observer_method):
        self.observers[observer_method.key] = observer_method
        self.evaluate_execution_order()

    def delete_observer(self, method):
        del self.observers[method.key]
        self.evaluate_execution_order()

    def evaluate_execution_order(self):
        keys = self.observers.keys()
        orders = [self.observers[key]['order'] for key in keys]
        self.execution_sort_idx = np.argsort(orders)

    def get_observers_in_execution_order(self):
        observers = [obs for obs in self.observers.values()]
        sort_idx = self.execution_sort_idx
        return [observers[p]['method'] for p in sort_idx]

    def dispatch_to_observers(self, *args, **kwargs):
        """
        Dispatch to observers. 

        Only rely on the order of execution if you have specified the 'order' argument for each observer. 
        """
        for observer_method in self.get_observers_in_execution_order():
            observer_method(*args, **kwargs)

    ####################################################################################################################
    # Printing / Reporting 
    ####################################################################################################################

    def print_observers(self, include_observer=False, verbose=0, hide_log=True):
        """
        include_observer flag might be useful to debug if something is not working as expected. 
        """

        order_indexs = self.execution_sort_idx
        keys = [key for key in self.observers.keys()]
        names = [obs['name'] for obs in self.observers.values()]
        methods = [obs['method'] for obs in self.observers.values()]
        orders = [obs['order'] for obs in self.observers.values()]
        
        base_string = '{}: order = {} - name = {} - method - {}'
        if include_observer:
            base_string += ' - method: {}'

        #start_string = '|'+'=' * 33 + ' Observers ' + '='*33+'|'
        header_print('Observers')
        for idx in order_indexs:
            
            if hide_log and 'LogEntry.' in names[idx]:
                continue

            pretty_print('  Order {} - Name: {}'.format(orders[idx], names[idx]))
            if verbose > 1:
                pretty_print('  Key: {}'.format(keys[idx]))
                pretty_print('  Method: {}'.format(methods[idx]))
                pretty_print('_'*50)

    def observer_reports(self, report_key=False, hide_log=True):

        dicts_out_of_order = [value for value in  self.observers.values()]

        
        header_print('Observers set/get reports')

        base_offset = '  '
        extra_offset = base_offset + '    '        
        for i in self.execution_sort_idx:

            observer_method = dicts_out_of_order[i]
            if 'LogEntry' in observer_method.name and hide_log:
                continue

            pretty_print(base_offset + observer_method.name)
            report = observer_method.report(offset=extra_offset, report_key=report_key, print_report=False, return_report=True)
            for string in report:
                pretty_print(string)
            else:
                #print(f"{dicts_out_of_order['name']} cannot report on its behavior!")
                pass

        get_set, set_set = self.get_set_match()
        pretty_print(base_offset)
        pretty_print(base_offset+'Overall:')
        pretty_print(base_offset+f'Get keys: {get_set}')
        pretty_print(base_offset+f'Set keys: {set_set}')
        pretty_print(base_offset+f'Key match: {get_set==set_set}')
        if not get_set==set_set:
            pretty_print(base_offset+'Sets do not match, this can be problematic!')
            if len(get_set) > len(set_set):
                pretty_print(base_offset+'Automatic check shows observers will attempt to get un-set item!')
                pretty_print(base_offset+'Program likely to crash!')
            if len(set_set) > len(get_set):
                pretty_print(base_offset+'Automatic check shows observers set value that is unused!')
                pretty_print(base_offset+'May cause unintended behaviour!')

            unmatched_keys = list(get_set.difference(set_set))+list(set_set.difference(get_set))
            pretty_print(base_offset+f'Umatched keys {unmatched_keys}')

    def get_set_match(self):
        dicts_out_of_order = [value for value in  self.observers.values()]
        all_sets = []
        all_gets = []

        for observer_method in dicts_out_of_order:
            all_sets += observer_method.sets.values()
            all_gets += observer_method.gets.values()

        all_sets = set(all_sets)
        all_gets = set(all_gets)

        return all_gets, all_sets
            
class FinalizationHandler:
    """
    Just stores information about functions to be called when finalizaing a run. 
    """

    def __init__(self):
        self.finalization_methods = {}
        self.names = {}

    def attach_finalization(self, name, method):
        key = method.__hash__()
        self.finalization_methods[key] = method
        self.names[key] = name
    
    def get_finalization_methods(self):
        return self.finalization_methods.values()

    def print_finalization(self, include_observer=False, verbose=0):
        """
        include_observer flag might be useful to debug if something is not working as expected. 
        """

        names = [self.names[key] for key in self.finalization_methods.keys()]
        
        base_string = '{}: order = {} - name = {} - method - {}'
        if include_observer:
            base_string += ' - method: {}'

        print('=' * 24 + ' Finalization ' + '='*24)
        for name in names:
            print('Name: {}'.format(name))
        print('='*len('=' * 25 + ' Observers ' + '='*25))
    
class Observer:

    """
    Base-class for classes that act as observers. 
    gets: dict
        Dict where the keys will be set as the name of attributes with the value being the value of the attribute. 
        Used to get something from the iteration_cache during a run. 

    sets: dict
        Dict where the keys will be set as the name of attributes with the value being the value of the attribute. 
        Used to set something from the iteration_cache during a run. 

    order: int/float
        Specifies the (relative) order of when the observer will be executed, lower numbers are executed first. 
    """

    def __init__(self, gets=[dict()], sets=[dict()], order=[0], sur_name='', **kwargs):
        """
        observer_dict: Dictionary where keys will become the name of attributes and values the value of said attribute. 
        So {'get_key':'candidate'} means self.get_key = 'candidate' which will then be used to access the iteration_cache
        dictionary during a run. 
        """

        if type(gets) == dict:
            gets = [gets]
        if type(sets) == dict:
            sets = [sets]
        if type(order) == int or type(order) == float:
            order = [order]

        combined = dict()
        for tuple_of_dicts in [gets, sets]:
            for dict_ in tuple_of_dicts:
                for key, item in dict_.items():
                    combined[key] = item
        #self.__dict__ = combined
        for key, value in combined.items():
            if key not in self.__dict__.keys():
                self.__dict__[key] = value
            else:
                raise BadOptionError('{}-set or get key has the same name as an attribute, this is not allowed'.format(key))

        self.set_keys = sum([list(set_dict.keys()) for set_dict in sets], [])
        self.set_values = sum([list(set_dict.values()) for set_dict in sets], [])
        self.get_keys = sum([list(get_dict.keys()) for get_dict in gets], [])
        self.get_values = sum([list(get_dict.values()) for get_dict in gets], [])
        self.gets = gets
        self.sets = sets

        self.order = order
        self.sur_name = sur_name

        self.observer_methods = {}

        if len(kwargs) > 0:
            print('Unused key word arguments supplied to {}'.format(self.name))
            for key, value in kwargs.items():
                print(key, value)

    @property
    def __name__(self):
        return self.name + self.sur_name

    def get_from_cache(self, key):
        """
        This should check according to the method that calls it.
        """
        assert key in self.get_values # Makes sure the module has said it wants to get with this key. 
        return self.main_get_from_cache(key)

    def add_to_cache(self, key, data, mode):
        """
        This should check according to the method that calls it.
        """
        assert key in self.set_values # Makes sure the module has said it wants to set with this key. 
        return self.main_add_to_cache(key, data, mode)

    def add_observer_method(self, method, sets, gets, order):        
        observer_method = ObserverMethod(self.__name__, method.__name__, method, gets, sets, order)
        self.observer_methods[observer_method.key] = observer_method

    def remove_observer_method(self, method):
        key = method.key
        if key in self.observer_methods.keys():
            del self.observer_methods[key]

    def update_order(self, method, order):
        key = method.key
        assert key in self.observer_methods.keys()
        self.observer_methods[key].order = order

    def attach(self, main):
        for observer_method in self.observer_methods.values():
            main.attach_observer(observer_method)

    def reset_observer(self):
        self.observer_methods = {}

    ####################################################################################################################
    # Misc. 
    ####################################################################################################################

    def assign_from_main(self, main):
        self.main_get_from_cache = main.get_from_cache
        self.main_add_to_cache = main.add_to_cache
        self.get_iteration_counter = main.get_iteration_counter
        self.candidate_instantiator = main.candidate_instantiator        

class ObserverMethod:

    def __init__(self, class_name, method_name, method, gets, sets, order):
        self.class_name = class_name
        self.method_name = method_name
        self.method = method
        self.gets = gets
        self.sets = sets
        self.order = order

        self.name = self.class_name + '.' + self.method_name
        self.class_reference = method.__self__
        #self.key = method.__hash__() # Old
        self.key = get_next_key()

    def __getitem__(self, key):
        return self.__dict__[key]

    def report(self, offset='', report_key=False, return_report=False, print_report=True):
        report = []

        for key in self.gets.keys():
            value = self.gets[key]
            out = offset + f"Gets '{value}'" 
            if report_key:
                out += f" using key attribute '{key}'"
            report.append(out)

        for key in self.sets.keys():
            value = self.sets[key]
            out = offset + f"Sets '{value}'"
            if report_key:
                out += f" using key attribute '{key}'"
            report.append(out)
        
        if len(self.gets) == 0 and len(self.sets) == 0:
            out = offset+'Doesnt set/get anything'
            report.append(out)

        if print_report:
            for string in report:
                print(string)
        if return_report:
            return report

if __name__ == '__main__':

    # class TestObserverObject:

    #     def __init__(self, name):
    #         self.name = name

    #     def test_observer_method(self):
    #         #print('Name {} - I am method {} on {}'.format(id(self.test_observer_method), id(self)))
    #         print('My name is {}'.format(self.name))

    # handler = ObserverHandler()

    # A = TestObserverObject('A')
    # B = TestObserverObject('B')

    # handler.add_observer(A.test_observer_method,'A', order=0)
    # handler.add_observer(B.test_observer_method,'B', order=0)
    # #handler.add_observer(B.test_observer_method,'B', order=-1)

    # for method in handler.get_observers_in_execution_order():
    #     method()

    #for key in handler.observers.keys():
    #    print(key, handler.observers[key])        

    # print(dir(B.test_observer_method))
    # print(B.test_observer_method.__self__)

    observer = Observer(get_key='candidates', set_key='candidates', bad_key='bad')
    
    print(observer.__dict__)

    observer.report()


        
    