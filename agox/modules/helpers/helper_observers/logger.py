import numpy as np
from agox.observer import Observer, ObserverHandler
from timeit import default_timer as dt
import pickle

from agox.modules.helpers.writer import header_footer, Writer

class Log(Writer):

    name = 'Log'

    def __init__(self):        
        Writer.__init__(self, verbose=True, use_counter=False)
        self.entries = {}

    def add_entry(self, observer_method):
        self.entries[observer_method.key] = LogEntry(observer_method)

    def __getitem__(self, item):
         return self.entries[item]

    def log_report(self):
        total_time = np.sum([entry.get_current_timing() for entry in self.entries.values()])
        self.writer('Total time {:05.2f} s '.format(total_time))

        for entry in self.entries.values():
            report = entry.get_iteration_report()
            for line in report:
                self.writer(line)

    def save_logs(self):
        with open('log.pckl', 'wb') as f:
            pickle.dump(self.entries, f)
    
    def restore_logs(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.entries = pickle.load(f)

    def plot_log(self, ax):
        for key, entry in self.entries.items():
            ax.plot(entry.timings, label=entry.name[10:])

        ax.set_xlim([0, len(entry.timings)])
        ax.legend()
        ax.set_xlabel('iteration [#]', fontsize=12)
        ax.set_ylabel('Timing [s]', fontsize=12)
        return ax

    def get_total_iteration_time(self):
        timings = []
        for entry in self.entries.values():
            timings.append(entry.timings)

        return np.mean(timings, axis=0)

class LogEntry(Observer, Writer):

    name = 'LogEntry'

    def __init__(self, observer_method):
        Observer.__init__(self, order=observer_method.order)
        Writer.__init__(self, verbose=True, use_counter=False)
        self.timings = []
        self.name = 'LogEntry.' + observer_method.name
        self.observer_name = observer_method.name

        # Time sub-entries:
        self.sub_entries = {}
        self.recursive_attach(observer_method)

    def attach(self, main):
        self.add_observer_method(self.start_timer, sets=self.sets[0], gets=self.gets[0], order=self.order[0]-0.01)
        self.add_observer_method(self.end_timer, sets=self.sets[0], gets=self.gets[0], order=self.order[0]+0.01)
        super().attach(main)

    def start_timer(self, *args, **kwargs):
        self.timings.append(-dt())

    def end_timer(self, *args, **kwargs):
        self.timings[-1] += dt()

    def get_current_timing(self):

        if len(self.timings):
            return self.timings[-1]
        else:
            print('Somehow the log failed - Havent figure out why this happens sometimes')
            return 0

    def get_sub_timings(self):
        if len(self.sub_entries):
            sub_timings = []
            for sub_entry in self.sub_entries.values():
                sub_timings.append(sub_entry.get_sub_timings())
            return sub_timings
        else:
            return self.get_current_timing()

    def recursive_attach(self, observer_method):

        if issubclass(observer_method.class_reference.__class__, ObserverHandler):
            
            # Want to get all other observers: 
            class_reference = observer_method.class_reference
            observer_dicts = observer_method.class_reference.observers

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
        self.sub_entries[observer_method.key] = LogEntry(observer_method)

    def get_iteration_report(self, report=None, offset=''):
        if report is None:
            report = [] # List of strings:

        report.append(offset + ' {} - {:05.2f} s'.format(self.observer_name, self.get_current_timing()))

        for sub_entry in self.sub_entries.values():
            report = sub_entry.get_iteration_report(report=report, offset=offset+' ' * 4)

        return report

class Logger(Observer, Writer):

    name = 'Logger'

    def __init__(self, save_frequency=100, **kwargs):        
        Observer.__init__(self, **kwargs)
        Writer.__init__(self, verbose=True, use_counter=False)
        self.log = Log()
        self.ordered_keys = []
        self.save_frequency = save_frequency

    def attach(self, main):
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
            self.log.add_entry(observer_method)
            self.log[key].attach(main)
    
        # Also attach a reporting:
        self.add_observer_method(self.report_logs, sets={}, gets={}, order=np.max(orders)+1)
        
        # Attach final dumping of logs:
        main.attach_finalization('Logger.dump', self.log.save_logs)

        super().attach(main)

    @header_footer
    def report_logs(self):
        self.log.log_report()
        
        if self.get_iteration_counter() % self.save_frequency == 0:
            self.log.save_logs()

