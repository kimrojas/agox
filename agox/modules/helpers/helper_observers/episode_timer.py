import numpy as np
from timeit import default_timer as dt

class EpisodeTimer:
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.timings = []        

    def start_timer(self):
        self.t0 = dt()

    def finish_timer(self):
        self.t1 = dt()
        self.timings.append(self.t1-self.t0)

        if self.verbose:
            print('Episode time: {}'.format(self.timings[-1]))

    def easy_attach(self, agox, start_order=-1, finish_order=5):
        agox.attach_observer('start_timer', self.start_timer, order=start_order)   
        agox.attach_observer('finish_timer', self.finish_timer, order=finish_order)

    def save_timings(self, file_name='episode_timings.npy'):
        np.save(file_name, self.timings)