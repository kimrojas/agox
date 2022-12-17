import numpy as np
from agox.observer import Observer
from agox.writer import Writer, agox_writer
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
from ase.io import write
from agox.candidates import StandardCandidate

class SamplableCandidatesRetriever(Observer, Writer):

    name = 'SamplableCandidatesRetriever'

    def __init__(self, tournament, mixed_database, sets={'set_key':'samplable_inner_candidates'}, gets={}, order=2.5):
        Observer.__init__(self, sets=sets, gets=gets, order=order)
        Writer.__init__(self)
        self.tournament = tournament
        self.mixed_database = mixed_database
        self.add_observer_method(self.manage_candidates, sets=self.sets[0], gets=self.gets[0], order=self.order[0])

    @agox_writer
    @Observer.observer_method
    def manage_candidates(self, state):

        player_id = state.get_iteration_counter() - 1
        self.writer('Player ID',player_id)

        player = self.tournament.get_player(player_id)
        if player.generation != 0:
            kmeans_selected_candidates_from_all_parents = self.mixed_database.get_all_candidates()
            for parent_id in [player.left.counter,player.right.counter]:

                # add the Kmeans-selected, local-model relaxed candidates from that parent
                # so that sampler in AGOX Generator has something to choose from
                kmeans_selected_candidates_from_this_parent = [candidate for candidate in kmeans_selected_candidates_from_all_parents if candidate.get_meta_information('player_id') == parent_id]
                copies = []
                for candidate in kmeans_selected_candidates_from_this_parent:
                    F = candidate.get_forces()
                    E = candidate.get_potential_energy()
                    candidate_copy = StandardCandidate(template=candidate.template,
                                                       numbers=candidate.get_atomic_numbers(),
                                                       positions=candidate.get_positions(),
                                                       cell=candidate.get_cell(),
                                                       pbc=candidate.get_pbc())
                    sp_calc = SinglePointCalculator(candidate_copy, energy=E, forces=F)
                    candidate_copy.set_calculator(sp_calc)
                    copies.append(candidate_copy)
                state.add_to_cache(self, self.set_key, copies, mode='a')


            
