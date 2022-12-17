from agox.evaluators.single_point import SinglePointEvaluator
from agox.observer import Observer
from agox.writer import Writer, agox_writer
from agox.candidates import StandardCandidate
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase import Atoms
import numpy as np

class ConditionalSinglePointEvaluator(SinglePointEvaluator):

    name = 'ConditionalSinglePointEvaluator'

    def __init__(self, calculator, mixed_database, **kwargs): 
        self.mixed_database = mixed_database
        super().__init__(calculator, **kwargs)
        self.E_lowest_DFT_energy = 1e5

    @agox_writer
    @Observer.observer_method
    def evaluate(self, state):
        
        player_id = state.get_iteration_counter() - 1
        self.writer('Player ID',player_id)

        self.writer(f'Lowest DFT energy: {self.E_lowest_DFT_energy:8.3f}')

        # get the candidates
        candidates = state.get_from_cache(self, self.get_key)
        done = False
        new_dft_data = []

        self.evaluated_candidates = []
        passed_evaluation_count = 0
        if self.do_check():
            while candidates and not done:
                self.writer(f'Trying candidate - remaining {len(candidates)}')
                candidate = candidates.pop(0)

                if candidate is None:
                    self.writer('Candidate was None - are your other modules working as intended?')
                    continue

                Emodel = candidate.get_potential_energy()
                if Emodel < self.E_lowest_DFT_energy:
                    do_dft_calculation = True
                elif Emodel < self.E_lowest_DFT_energy + 1:
                    do_dft_calculation = np.random.rand() < 0.1
                else:
                    do_dft_calculation = np.random.rand() < 0.02
                if do_dft_calculation:
                    internal_state = self.evaluate_candidate(candidate)
                else:
                    internal_state = True

                if internal_state:
                    self.writer('Succesful calculation of candidate.')
                    passed_evaluation_count += 1

                    E = candidate.get_potential_energy()
                    if do_dft_calculation:
                        print(f'Structure with model energy: {Emodel:8.3f} and DFT energy: {E:8.3f}')
                        if E < self.E_lowest_DFT_energy:
                            self.E_lowest_DFT_energy = E
                    else:
                        print(f'Structure with model energy: {Emodel:8.3f}')
                    F = candidate.get_forces()
                    candidate_copy = StandardCandidate(template=candidate.template,
                                                       numbers=candidate.get_atomic_numbers(),
                                                       positions=candidate.get_positions(),
                                                       cell=candidate.get_cell(),
                                                       pbc=candidate.get_pbc())
                    sp_calc = SinglePointCalculator(candidate_copy, energy=E, forces=F)
                    candidate_copy.set_calculator(sp_calc)
                    candidate_copy.add_meta_information('player_id', player_id)
                    self.mixed_database.store_candidate(candidate_copy, accepted=True, write=True, dispatch=False)

                    if do_dft_calculation:
                        self.evaluated_candidates[-1].add_meta_information('final', True)
                        a = Atoms(candidate.get_atomic_numbers(), positions=candidate.get_positions(), cell=candidate.get_cell(), pbc=candidate.get_pbc())
                        sp_calc = SinglePointCalculator(a, energy=E, forces=F)
                        a.set_calculator(sp_calc)
                        new_dft_data.append(a)

                    if passed_evaluation_count == self.number_to_evaluate:
                        self.writer('Calculated required number of candidates.')
                        done = True
        
        state.add_to_cache(self, self.set_key, self.evaluated_candidates, mode='a')
        write(f'new_dft_data_{player_id:04d}.traj',new_dft_data)
