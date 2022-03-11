import numpy as np
from agox.observer_handler import Observer
from timeit import default_timer as dt

class Timer(Observer):
    name = 'Timer'
    def __init__(self, verbose=True, start_order=0, finish_order=10):
        super().__init__()
        self.verbose = verbose
        self.timings = []
        self.start_order = start_order
        self.finish_order = finish_order

    def start_timer(self):
        self.t0 = dt()

    def finish_timer(self):
        self.t1 = dt()
        self.timings.append(self.t1-self.t0)

        if self.verbose:
            print('Time: {}'.format(self.timings[-1]))

    def attach(self, agox):
        agox.attach_observer('start_timer', self.start_timer, order=self.start_order)   
        agox.attach_observer('finish_timer', self.finish_timer, order=self.finish_order)

    def save_timings(self, file_name='episode_timings.npy'):
        np.save(file_name, self.timings)