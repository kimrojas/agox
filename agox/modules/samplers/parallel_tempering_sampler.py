import os
from time import sleep
import numpy as np
from ase.io import write, read
from agox.modules.samplers.metropolis import MetropolisSampler

from ase.calculators.singlepoint import SinglePointCalculator 


def safe_write(filename, atoms):
    write(filename=filename+'.lock', images=atoms, format='traj')
    os.rename(filename+'.lock', filename)

class ParallelTemperingSampler(MetropolisSampler):
    """
    Written to work with ConcurrentDatabase, no guarantee that it works in any other setting. 
    """
    
    name = 'ParallelTemperingSampler'

    def __init__(self, temperatures=[], swap_frequency=10, gets=[{'get_key':'evaluated_candidates'}, {}], 
                sets=[{'set_key':'evaluated_candidates'}, {}], swap_order=8, **kwargs):
        super().__init__(gets=gets, sets=sets, **kwargs)
        self.swap_frequency = swap_frequency
        self.temperatures = temperatures
        self.temperature = temperatures[self.database.worker_number]
        self.swap_order=swap_order
        self.swap_func = self.metropolis_hastings_swap

        self.add_observer_method(self.swap_candidates, sets=self.sets[1], gets=self.gets[1], order=self.swap_order)

    def swap_candidates(self):
        """
        Assumes that the database is synced. 
        """
        
        if self.decide_to_swap():

            most_recently_accepted = {worker_number:None for worker_number in range(self.database.total_workers)}
            most_recently_accepted_iteration = {worker_number:0 for worker_number in range(self.database.total_workers)}
            for candidate in self.database.candidates:

                worker_number = candidate.get_meta_information('worker_number')
                iteration = candidate.get_meta_information('iteration')
                accepted = candidate.get_meta_information('accepted')

                if accepted:
                    if iteration > most_recently_accepted_iteration[worker_number]:
                        most_recently_accepted[worker_number] = candidate
                        most_recently_accepted_iteration[worker_number] = candidate.get_meta_information('iteration')

            self.swap_func(most_recently_accepted)

    def swap_down(self, most_recently_accepted):
        """
        Swaps candidates 'down' in worker_number:

        So if 4 total workers: 
        worker 0: Gets from worker 1.
        worker 1: Gets from worker 2. 
        worker 2: Gets from worker 3.
        worker 3: Gets from worker 4. 
        """
        
        total_workers = self.database.total_workers
        worker_number = self.database.worker_number
        if worker_number < total_workers-1:
            self.chosen_candidate = most_recently_accepted[worker_number+1]
        else:
            self.chosen_candidate = most_recently_accepted[total_workers-1]

        self.writer('Finish swapping candidate!')

    def metropolis_hastings_swap(self, most_recently_accepted):
        worker_number = self.database.worker_number
        total_workers = self.database.total_workers
        iteration = self.get_iteration_counter()

        # I abuse the filenames a bit here: 
        filename = self.database.filename[:-3] + '_swap_iteration_{}_worker_{}.traj'

        if worker_number == 0:
            # This one does the calculation and 'broadcasts' to the others over disk. 

            # Starting from the bottom: 
            for i in range(total_workers-1):                
                C_i = most_recently_accepted[i]
                C_j = most_recently_accepted[i+1]

                E_i = C_i.get_potential_energy()
                E_j = C_j.get_potential_energy()

                beta_i = 1/self.temperatures[i]
                beta_j = 1/self.temperatures[i+1]

                P = np.min([1, np.exp((beta_i-beta_j)*(E_i-E_j))])

                r = np.random.rand()

                if r < P:
                    self.writer('Swapped {} with {}'.format(i, i+1))
                    most_recently_accepted[i], most_recently_accepted[i+1] = most_recently_accepted[i+1], most_recently_accepted[i]
                else:
                    self.writer('Did not swap {} for {}'.format(i, i+1))

            # Write the candidates:
            for wn in range(1, total_workers):
                safe_write(filename.format(iteration, wn), most_recently_accepted[wn])
            self.chosen_candidate = most_recently_accepted[worker_number]

        else: 
            while not os.path.exists(filename.format(iteration, worker_number)):
                sleep(1)
            
            chosen_atoms = read(filename.format(iteration, worker_number))
            
            self.chosen_candidate = self.convert_to_candidate_object(chosen_atoms, self.chosen_candidate.template)
            scp = SinglePointCalculator(self.chosen_candidate, energy=chosen_atoms.get_potential_energy(), forces=chosen_atoms.get_forces())
            self.chosen_candidate.set_calculator(scp)

            os.remove(filename.format(iteration, worker_number))

    def convert_to_candidate_object(self, atoms_type_object, template):
        candidate =  self.candidate_instantiator(template=template, positions=atoms_type_object.positions, numbers=atoms_type_object.numbers, 
                                          cell=atoms_type_object.cell)
        return candidate

    def decide_to_swap(self):
        return (self.get_iteration_counter() % self.swap_frequency == 0) * (self.database.total_workers > 1)

    def get_candidate_to_consider(self):        
        candidates = self.get_from_cache(self.get_key)

        if not len(candidates) > 0:
            return None
        return candidates[0]

    def setup(self):
        candidate = super().setup()
        # Why do I do this??? 
        self.add_to_cache(self.set_key, [candidate], mode='w')

    def assign_from_main(self, main):
        self.candidate_instantiator = main.candidate_instantiator
        return super().assign_from_main(main)
    