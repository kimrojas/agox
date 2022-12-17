import numpy as np
from agox.observer import Observer
from agox.writer import Writer, agox_writer
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
from ase.io import read

class NonSamplableCandidatesRetriever(Observer, Writer):

    name = 'NonSamplableCandidatesRetriever'

    def __init__(self, tournament, run_idx, sets={'set_key':'nonsamplable_inner_candidates'}, gets={}, order=2.5, latest_N=300):
        Observer.__init__(self, sets=sets, gets=gets, order=order)
        Writer.__init__(self)
        self.tournament = tournament
        self.run_idx = run_idx
        self.add_observer_method(self.manage_inner_candidates, sets=self.sets[0], gets=self.gets[0], order=self.order[0])
        self.latest_N = latest_N

    @agox_writer
    @Observer.observer_method
    def manage_inner_candidates(self, state):

        player_id = state.get_iteration_counter() - 1
        self.writer('Player ID',player_id)
            
        player = self.tournament.get_player(player_id)
        if player.generation != 0:
            for parent_id in [player.left.counter,player.right.counter]:
                # add the latest N structures from that parent to the inner database so that
                # the global-GPR has some stuff to start from
                filename = f'dumped_candidates_run{self.run_idx:04d}_restart{parent_id:04d}.traj'
                candidates_from_parent = read(filename,index=':')
                if len(candidates_from_parent) > self.latest_N:
                    candidates_from_parent = candidates_from_parent[-self.latest_N:]
                state.add_to_cache(self, self.set_key, candidates_from_parent, mode='a')
