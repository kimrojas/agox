import numpy as np
from abc import ABC, abstractmethod

class ObserverHandler:
    """
    Base-class for classes that can have attached observers. 
    """

    def __init__(self):
        self.observers = {}
        self.execution_sort_idx = []

    def attach_observer(self, name, method, order=0):
        key = method.__hash__()

        if hasattr(method, '__self__'):
            class_reference = method.__self__
        else:
            class_reference = None

        self.observers[key] = {'method':method, 'name':name, 'order':order, 'class_reference':class_reference}
        
        self.evaluate_execution_order()

    def delete_observer(self, method):
        del self.observers[method.__hash__()]
        self.evaluate_execution_order()

    def evaluate_execution_order(self):
        keys = self.observers.keys()
        orders = [self.observers[key]['order'] for key in keys]
        self.execution_sort_idx = np.argsort(orders)

    def get_observers_in_execution_order(self):
        observers = [obs for obs in self.observers.values()]
        sort_idx = self.execution_sort_idx
        return [observers[p]['method'] for p in sort_idx]

    def print_observers(self, include_observer=False, verbose=0):
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

        start_string = '|'+'=' * 35 + ' Observers ' + '='*35+'|'
        self.total_print_length = len(start_string)
        end_string = '|'+'=' * (len(start_string)-2)+'|'
        print(start_string)
        for idx in order_indexs:
            self.pretty_print('|  Order {} - Name: {}'.format(orders[idx], names[idx]))
            if verbose > 1:
                self.pretty_print('|  Key: {}'.format(keys[idx]))
                self.pretty_print('|  Method: {}'.format(methods[idx]))
                self.pretty_print('_'*50)
        print(end_string)
    
    def dispatch_to_observers(self, *args, **kwargs):
        """
        Dispatch to observers. 

        Only rely on the order of execution if you have specified the 'order' argument for each observer. 
        """
        for observer_method in self.get_observers_in_execution_order():
            observer_method(*args, **kwargs)

    def observer_reports(self, report_key=False):

        dicts_out_of_order = [value for value in  self.observers.values()]

        start_string = '|'+'='*27 + ' Observers set/get reports ' + '='* 27+'|'
        self.total_print_length = len(start_string)
        end_string = '|'+ '=' * (len(start_string)-2) + '|'
        
        print(start_string)

        base_offset = '|  '
        extra_offset = base_offset + '    '        
        for i in self.execution_sort_idx:

            class_reference = dicts_out_of_order[i]['class_reference']
            if class_reference is not None:
                if hasattr(class_reference, 'report'):
                    self.pretty_print(base_offset + class_reference.name)
                    report = class_reference.report(offset=extra_offset, report_key=report_key, print_report=False, return_report=True)
                    for string in report:
                        self.pretty_print(string)
            else:
                print(f"{dicts_out_of_order['name']} cannot report on its behavior!")

        get_set, set_set = self.get_set_match()
        self.pretty_print(base_offset)
        self.pretty_print(base_offset+'Overall:')
        self.pretty_print(base_offset+f'Get keys: {get_set}')
        self.pretty_print(base_offset+f'Set keys: {set_set}')
        self.pretty_print(base_offset+f'Key match: {get_set==set_set}')
        if not get_set==set_set:
            self.pretty_print(base_offset+'Sets do not match, this can be problematic!')
            if len(get_set) > len(set_set):
                self.pretty_print(base_offset+'Automatic check shows observers will attempt to get un-set item!')
                self.pretty_print(base_offset+'Program likely to crash!')
            if len(set_set) > len(get_set):
                self.pretty_print(base_offset+'Automatic check shows observers set value that is unused!')
                self.pretty_print(base_offset+'May cause unintended behaviour!')

            unmatched_keys = list(get_set.difference(set_set))+list(set_set.difference(get_set))
            self.pretty_print(base_offset+f'Umatched keys {unmatched_keys}')

        print(end_string)

    def get_set_match(self):
        dicts_out_of_order = [value for value in  self.observers.values()]


        all_sets = []
        all_gets = []

        for observer_dict in dicts_out_of_order:
            class_reference = observer_dict['class_reference']

            if class_reference is not None: 
                all_sets += class_reference.set_values
                all_gets += class_reference.get_values

        all_sets = set(all_sets)
        all_gets = set(all_gets)

        return all_gets, all_sets
        
    def pretty_print(self, string):
        string += (self.total_print_length - len(string)-1) * ' ' + '|'
        print(string)
            
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
        Used to get something from the episode_cache during a run. 

    sets: dict
        Dict where the keys will be set as the name of attributes with the value being the value of the attribute. 
        Used to set something from the episode_cache during a run. 

    order: int/float
        Specifies the (relative) order of when the observer will be executed, lower numbers are executed first. 
    """

    def __init__(self, gets=dict(), sets=dict(), order=0):
        """
        observer_dict: Dictionary where keys will become the name of attributes and values the value of said attribute. 
        So {'get_key':'candidate'} means self.get_key = 'candidate' which will then be used to access the episode_cache
        dictionary during a run. 
        """    
        combined = dict()
        for dict_ in [gets, sets]:
            for key, item in dict_.items():
                combined[key] = item
        self.__dict__ = combined
        self.set_keys = list(sets.keys())
        self.set_values = list(sets.values())
        self.get_keys = list(gets.keys())
        self.get_values = list(gets.values())
        self.order = order

    def report(self, offset='', report_key=False, return_report=False, print_report=True):

        report = []

        for key in self.get_keys:
            value = self.__dict__[key]
            out = offset + f"Gets '{value}'" 
            if report_key:
                out += f" using key attribute '{key}'"
            report.append(out)

        for key in self.set_keys:
            value = self.__dict__[key]
            out = offset + f"Sets '{value}'"
            if report_key:
                out += f" using key attribute '{key}'"
            report.append(out)
        
        if len(self.get_keys) == 0 and len(self.set_keys) == 0:
            out = offset+'Doesnt set/get anything'
            report.append(out)

        if print_report:
            for string in report:
                print(string)
        if return_report:
            return report

    def get_from_cache(self, key):
        assert key in self.get_values # Makes sure the module has said it wants to get with this key. 
        return self.main_get_from_cache(key)

    def add_to_cache(self, key, data, mode):
        assert key in self.set_values # Makes sure the module has said it wants to set with this key. 
        return self.main_add_to_cache(key, data, mode)

    def assign_from_main(self, main):
        self.main_get_from_cache = main.get_from_cache
        self.main_add_to_cache = main.add_to_cache
        self.get_episode_counter = main.get_episode_counter
        self.candidate_instantiator = main.candidate_instantiator        

    def simple_attach(self, main, name, func):
        self.assign_from_main(main)
        main.attach_observer(name, func, order=self.order)

    def attach(self, main):
        pass

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


        
    