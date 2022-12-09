import numpy as np
from agox.observer import Observer
from agox.writer import Writer, agox_writer
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
from ase.io import write

class CandidateDumper(Observer, Writer):

    name = 'CandidateDumper'

    def __init__(self, filename, sets={}, gets={'get_key':'candidates'}, order=2.5):
        Observer.__init__(self, sets=sets, gets=gets, order=order)
        Writer.__init__(self)
        self.filename = filename
        self.add_observer_method(self.dump_candidates, sets=self.sets[0], gets=self.gets[0], order=self.order[0])

    @agox_writer
    @Observer.observer_method
    def dump_candidates(self, state):

        self.writer('Doing stuff')

        atoms = []
        candidates = state.get_from_cache(self, self.get_key)
        for i,cached_candidate in enumerate(candidates):
            E = cached_candidate.get_potential_energy()
            F = cached_candidate.get_forces()
            a = Atoms(cached_candidate.get_atomic_numbers(),
                      positions=cached_candidate.get_positions(),
                      cell=cached_candidate.get_cell(),
                      pbc=cached_candidate.get_pbc())
            sp_calc = SinglePointCalculator(a, energy=E, forces=F)
            a.set_calculator(sp_calc)
            atoms.append(a)
        write(self.filename,atoms)

        #state.add_to_cache(self, self.set_key, candidates_with_model_energies, mode='a')

            
