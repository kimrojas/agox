from agox.writer import Writer, agox_writer
from agox.observer import Observer
from ase.io import read, write
import os

class CandidateTracker(Observer, Writer):

    name = 'CandidateTracker'

    def __init__(self, get_key, order=[3], save_path='', save_name='', save=False):
        gets = [{'get_key':get_key}]
        Observer.__init__(self, gets=gets, order=order)
        Writer.__init__(self)
        self.add_observer_method(self.track_candidates, order=self.order[0], 
            sets=self.sets[0], gets=self.gets[0])

        self.save = save
        self.save_path = save_path
        self.save_name = save_name

    @agox_writer
    @Observer.observer_method
    def track_candidates(self, state):

        candidates = state.get_from_cache(self, self.get_key)

        for candidate in candidates:
            E = candidate.get_potential_energy()
            self.writer(f'Candidate with energy {E}')

        self.save_candidates(candidates)

    def save_candidates(self, candidates):
        if self.save:
            path = os.path.join(self.save_path, f'{self.save_name}_{self.get_iteration_counter()}.traj')
            write(path, candidates)

