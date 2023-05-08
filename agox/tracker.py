import numpy as np
from agox.observer import Observer, ObserverHandler
from agox.writer import Writer, agox_writer
from agox.module import Module
from timeit import default_timer as dt
from agox.databases import DatabaseBaseClass

class Tracker(Observer, Writer):
    """
    Logger that looks at modules and logs properties they have assigned each iteration. 
    Also is responsible for timings. 
    """

    name = 'Tracker'

    def __init__(self, agox, save_name=None):
        order = agox.get_max_order() + 2
        all_gets, all_sets = agox.get_set_match()
        gets = {key:key for key in all_gets}
        Observer.__init__(self, order=order, gets=gets)
        Writer.__init__(self, verbose=True, use_counter=False, prefix='')

        self.tracked_attributes = {}
        self.state_loggers = []

        self.timer = Timer()
        self.timer.attach(agox)

        self.find_modules(agox)
        self.find_trackable_attributes()
        self.add_observer_method(self.update, sets={}, gets=gets, order=order)
        self.attach(agox)

        # Naming for files:
        database_file_name = self.find_database_name()
        if save_name is None:
            self.save_name = database_file_name + '_tracker'
        else:
            self.save_name = save_name

        agox.attach_finalization('tracker_save', self.save)

    def find_modules(self, agox):
        modules = [obs.class_reference for obs in agox.observers.values()]
        outer_modules = {element.cache_key:element for element in modules}

        all_modules = {key:val for key, val in outer_modules.items()}

        for module in outer_modules.values():
            submodules = module.find_submodules(only_dynamic=False)

            for submodule in submodules.values():
                all_modules[submodule.cache_key] = submodule

        self.modules_to_track = all_modules

    def find_database_name(self):
        database_filename = 'tracker'            
        for module in self.modules_to_track.values():
            if issubclass(module.__class__, DatabaseBaseClass):
                if hasattr(module, 'filename'):
                    database_filename = module.filename
                    # Without extension:
                    database_filename = database_filename.split('.')[0]
                    break
        return database_filename
            
    def get_attribute_name(self, module, attribute_key):
        return f'{module.__name__}.{attribute_key}'

    def find_trackable_attributes(self):
        for key, module in self.modules_to_track.items():
            for attribute_key in module.tracked_attributes:
                attribute_name = self.get_attribute_name(module, attribute_key)
                self.tracked_attributes[attribute_name] = []

        self.tracked_attributes['Timer.tracker.update'] = []


    @agox_writer
    @Observer.observer_method
    def update(self, state):
        t0 = dt()

        for module in self.modules_to_track.values():
            attribute_dict = module.get_tracked_attributes()
            for keys, tracked_attribute in attribute_dict.items():
                attribute_name = self.get_attribute_name(module, keys)
                self.tracked_attributes[attribute_name].append(tracked_attribute)

        for state_logger_tuple in self.state_loggers:
            function, name, gets, tracking_name = state_logger_tuple
            self.tracked_attributes[tracking_name].append(function(state.get_from_cache(self, gets)))

        t1 = dt()
        self.tracked_attributes['Timer.tracker.update'].append(t1-t0)

        
    def add_state_tracker(self, function, name, gets):
        tracking_name = f'state.{gets}.{name}'
        self.state_loggers.append((function, name, gets, tracking_name))
        self.tracked_attributes[tracking_name] = []

    def save(self):       
        np.savez(self.save_name+'.npz', **self.tracked_attributes)

    @staticmethod
    def load(filename):
        return {key:value for key, value in np.load(filename).items()}

class Timer(Observer, Writer):

    name = 'Timer'

    def __init__(self, save_frequency=100, **kwargs):
        """
        Logger instantiation. 

        The 'Logger' is the observer that is attached to main and prints 
        the log report. 

        Parameters:
        save_frequency: int
            How often to save to save the log to disk.
        """

        Observer.__init__(self, **kwargs)
        Writer.__init__(self, verbose=True, use_counter=False)
        self.timers = []
        self.ordered_keys = []
        self.save_frequency = save_frequency

        self.add_tracked_attribute('iteration_time')

    def attach(self, main):
        """
        Attaches to main. 

        Three things happen:
        1.  LogEntry's are created for each observer, which may also recursively 
            create LogEntry's for observers of observers and so on.
        2.  The 'Logger' attaches an 'report_logs' as an Observer to main, so 
            that the log is printed. 
        3.  A finalization method is added so that the log is saved when AGOX 
            finishes. 

        Parameters
        -----------
        main: AGOX object.
            The main AGOX class from agox/main.py
        """
        # Want to get all other observers: 
        observers = main.observers

        # Understand their order of execution:
        keys = []; orders = []
        for observer_method in observers.values():
            keys.append(observer_method.key) 
            orders.append(observer_method.order)
        sort_index = np.argsort(orders)
        sorted_keys = [keys[index] for index in sort_index]
        sorted_observers = [observers[key] for key in sorted_keys]

        # Attach log obserers to time them:
        for observer_method, key in zip(sorted_observers, sorted_keys):
            observer_timer = TimerEntry(observer_method)
            observer_timer.attach(main)
            self.timers.append(observer_timer)
    
        # Also attach a reporting:
        self.add_observer_method(self.report_logs, sets={}, gets={}, order=np.max(orders)+1)
        
        # Attach final dumping of logs:
        #main.attach_finalization('Logger.dump', self.log.save_logs)

        super().attach(main)

    @agox_writer
    @Observer.observer_method
    def report_logs(self, state):
        """
        Calls the 'log_report' of the log object.

        Saves the log to disk depending on self.save_frequency. 
        """
        total_time = np.sum([entry.get_current_timing() for entry in self.timers])
        self.iteration_time = total_time
        self.writer('Total time {:05.2f} s '.format(total_time))

        for entry in self.timers:
            report = entry.get_iteration_report()
            for line in report:
                self.writer(line)
        
class TimerEntry(Observer, Writer, Module):

    name = 'TimerEntry'

    def __init__(self, observer_method):
        """
        TimerEntry class. 
        
        Given an ObserverMethod the LogEntry attaches observers around that method 
        in the observer-loop to time the ObserverMethod. 

        If the Observer that ObserverMethod comes from is an instance of an 
        ObserverHandler it rescursively attaches other LogEntry instances to 
        the observers of that ObserverHandler to time those too. 

        Parameters
        -----------
        observer_method: ObserverMethod object. 
            An instance of a ObserverMethod (agox/observer.py) to attach around. 
        """

        Observer.__init__(self, order=observer_method.order)
        Writer.__init__(self, verbose=True, use_counter=False)
        self.timings = []
        self.name = 'Timer.' + observer_method.name
        self.observer_name = observer_method.name

        # Time sub-entries:
        self.sub_entries = {}
        self.recursive_attach(observer_method)

        self.add_tracked_attribute('timing')

    def attach(self, main):
        """
        Attachs class methods to the observer loop of main. 

        Parameters
        -----------
        main: AGOX object.
            The main AGOX class from agox/main.py
        """

        self.add_observer_method(self.start_timer, sets=self.sets[0], gets=self.gets[0], order=self.order[0]-0.01)
        self.add_observer_method(self.end_timer, sets=self.sets[0], gets=self.gets[0], order=self.order[0]+0.01)
        super().attach(main)

    def start_timer(self, state, *args, **kwargs):
        """
        Method attached as an observer to start the timing. 
        """
        self.timings.append(-dt())
        
    def end_timer(self, state, *args, **kwargs):
        """
        Method attached as an observer to end the timing.
        """
        if len(self.timings):
            self.timings[-1] += dt()

        self.timing = self.timings[-1]

    def get_current_timing(self):
        """
        Get most recent timing. 

        Returns
        --------
        float
            Timing of most recent iteration.
        """

        if len(self.timings):
            return self.timings[-1]
        else:
            print(f'{self.name}: Somehow the timer failed - Havent figure out why this happens sometimes - {len(self.timings)}')
            return 0

    def get_sub_timings(self):
        """
        Get timing of sub entries, relevant when the object self is attached around 
        is also an ObserverHandler that has other Observers, such as a Database 
        that may have a model listening to it. 

        Returns
        --------
        List or float
            If sub_entries are present then returns a list of those subtimings 
            otherwise returns a throguh 'get_current_timing'.
        """
        if len(self.sub_entries):
            sub_timings = []
            for sub_entry in self.sub_entries.values():
                sub_timings.append(sub_entry.get_sub_timings())
            return sub_timings
        else:
            return self.get_current_timing()

    def recursive_attach(self, observer_method):
        """
        Attach recursively when the class that ObserverMethod originates from 
        is an instance of ObserverHandler.

        Parameters
        -----------
        observer_method: ObserverMethod object. 
            An instance of a ObserverMethod (agox/observer.py) self is attached 
            around. 
        """
        if issubclass(observer_method.class_reference.__class__, ObserverHandler):

            if not observer_method.name == observer_method.class_reference.dispatch_method:
                return
            
            # Want to get all other observers: 
            class_reference = observer_method.class_reference # The class the inbound observer method comes from.
            observer_dicts = observer_method.class_reference.observers # Dict of observer methods.


            # May sometimes find objects that already have LogEntry observers 
            # attached, so we remove those. 
            methods_to_delete = []
            for key, observer_method in observer_dicts.items():
                if 'LogEntry' in observer_method.name:
                    methods_to_delete.append(observer_method)

            for method in methods_to_delete:
                class_reference.delete_observer(method)                    

            # Understand their order of execution:
            keys = []; orders = []
            for key, observer_dict in observer_dicts.items():
                keys.append(key) 
                orders.append(observer_dict['order'])
            sort_index = np.argsort(orders)
            sorted_keys = [keys[index] for index in sort_index]
            sorted_observer_dicts = [observer_dicts[key] for key in sorted_keys]

            # Attach log entry observers to time them:
            for observer_method, key in zip(sorted_observer_dicts, sorted_keys):
                self.add_sub_entry(observer_method)
                self.sub_entries[observer_method.key].attach(class_reference)

    def add_sub_entry(self, observer_method):
        """
        Add a sub entry which is an other instance of LogEntry. 

        Parameters
        -----------
        observer_method: ObserverMethod object. 
            An instance of a ObserverMethod (agox/observer.py) from an Observer 
            that is observing the Observer self is attached around. 
            E.g. if self is attached around a Database then a Model may be 
            observing that database with an ObserverMethod.  
        """
        self.sub_entries[observer_method.key] = TimerEntry(observer_method)

    def get_iteration_report(self, report=None, offset=''):
        """
        Recursively generate iteration report. 

        Parameter
        ----------
        report: None or a list. 
            If None an empty list is created. Report strings are appended to that
            list.
        offset: str, 
            Indentation, reports from subentries are offset by 4 spaces. 

        Returns
        --------
        list
            The 'report' list of strings is returned. 
        """
        if report is None:
            report = [] # List of strings:

        report.append(offset + ' {} - {:05.2f} s'.format(self.observer_name, self.get_current_timing()))

        for sub_entry in self.sub_entries.values():
            report = sub_entry.get_iteration_report(report=report, offset=offset+' ' * 4)

        return report