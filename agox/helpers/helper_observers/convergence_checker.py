import numpy as np
from agox.observer import Observer
from agox.writer import Writer, agox_writer

class ConvergenceChecker(Observer, Writer):

    name = 'ConvergenceChecker'

    def __init__(self, database=None, improvement_threshold=0.0, stagnation_threshold=5, gets={'get_key':'evaluated_candidates'}, sets={}, order=0, 
                verbose=True, use_counter=True, start_iteration=0, prefix='', **kwargs):
        Observer.__init__(self, sets=sets, gets=gets, order=order)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)

        self.database = database
        self.improvement_threshold = improvement_threshold        
        self.stagnation_counter = 0
        self.stagnation_threshold = stagnation_threshold
        self.start_iteration = start_iteration

        self.add_observer_method(self.check_for_convergence, sets=self.sets[0], gets=self.gets[0], order=self.order[0])

    def check_for_convergence(self):
        if self.get_iteration_counter() < self.start_iteration:
            return False

        # Get best energy from database: 
        best_energy = self.database.get_best_energy(return_preset=False)

        # Get candidates from current iteration: 
        candidates = self.get_from_cache(self.get_key)
        candidate_energies = [candidate.get_potential_energy() for candidate in candidates]
        best_candidate_energy = np.min(candidate_energies)

        improved = best_candidate_energy < best_energy - self.improvement_threshold

        if not improved:
            self.stagnation_counter += 1
            return self.stagnation_counter >= self.stagnation_threshold
        else:
            self.stagnation_counter = 0
            return False

class ConvergenceCheckerRelaxUpdate(ConvergenceChecker):

    def __init__(self, evaluator, multiplier=10, **kwargs):
        super().__init__(**kwargs)
        self.evaluator_instance = evaluator
        self.default_number_of_steps = self.evaluator_instance.optimizer_run_kwargs['steps'] # This is not a reference
        self.multiplier = multiplier
        self.relaxer_updated = False

    def update_evaluator_steps(self):
        self.evaluator_instance.optimizer_run_kwargs['steps'] = int(self.multiplier * self.default_number_of_steps)
        self.relaxer_updated = True

    def reset_evaluator_steps(self):
        self.evaluator_instance.optimizer_run_kwargs['steps'] = self.default_number_of_steps
        self.relaxer_updated = False

    def check_for_convergence(self):
         # Get best energy from database: 
        best_energy = self.database.get_best_energy(return_preset=False)

        # Get candidates from current iteration: 
        candidates = self.get_from_cache(self.get_key)
        candidate_energies = [candidate.get_potential_energy() for candidate in candidates]
        best_candidate_energy = np.min(candidate_energies)

        improved = best_candidate_energy < best_energy - self.improvement_threshold

        if not improved:
            self.stagnation_counter += 1
            state = self.stagnation_counter >= self.stagnation_threshold
        else:
            self.stagnation_counter = 0
            state = False

        if self.stagnation_counter < self.stagnation_threshold - 3 and not self.relaxer_updated:
            self.update_evaluator_steps()

        if best_candidate_energy < best_energy - self.improvement_threshold and self.relaxer_updated:
            self.reset_evaluator_steps()

        return state

class ConvergenceCheckerBjorkMode(ConvergenceChecker):

    def __init__(self, sampler, improvement_threshold=0.5, **kwargs):
        super().__init__(improvement_threshold=improvement_threshold, sets={'set_key':'sampled_candidates'}, **kwargs)
        self.best_energy = 10E10
        self.begin_change = 3
        self.sampler = sampler

    @agox_writer
    def check_for_convergence(self):

        if self.get_iteration_counter() < self.start_iteration:
            self.writer(f'Not starting yet: {self.get_iteration_counter()}/{self.start_iteration}')
            return False

        print(f'Running convergence checker {self.get_iteration_counter()}/{self.start_iteration}')

        # Step 1.
        energy = self.database.get_best_energy(return_preset=False)

        # Step 2. 
        candidates = self.get_from_cache(self.get_key)
        energies = [c.get_potential_energy() for c in candidates]
        # Step 3.
        energy = np.min([np.min(energies), energy])

        # Step 4.
        delta_energy = energy - self.best_energy

        # Step 5:
        if delta_energy < 0: 
            self.best_energy = energy
            self.writer('Candidate marginally better than best previous!')
        # Step 6:
        if delta_energy <= -self.improvement_threshold:
            self.writer('Candidate better than improvement threshold!')
            self.stagnation_counter = 0
            return False
        # Step 7
        self.stagnation_counter += 1
        state = (self.stagnation_counter >= self.stagnation_threshold)

        self.writer(f'Stagnation counter = {self.stagnation_counter} out of {self.stagnation_threshold}')

        # Step 8:
        if self.stagnation_counter >= (self.stagnation_threshold - self.begin_change):

            # Step 9 Get candidates from sampler:
            sampled_candidates = self.sampler.get_all_members()
            self.add_to_cache(self.set_key, sampled_candidates, mode='w')
            self.writer('Added {} candidates from the sampler!'.format(len(sampled_candidates)))

        if state:
            self.writer('Convergence detected AGOX should stop now!')

        return state
        
# Bjørk strategy:
# 1. Get best energy from database --> energy
# 2. Get candidates from cache
# 3. Compare energies from candidates to energy from database. 
#    If better set to --> energy 
# 4. Calculate delta_energy = energy - self.best_energy
# 5. If delta_energy < 0 --> self.best_energy = energy
# 6. If dalta_energy < -0.5 --> Reset -> stalled iterations = 0 and return
# 7. Increase number of stalled iterations. 
# 8. If stalled_iterations >= max_stalled_iterations 
# 9. Adds candidates from sample to cache with key = 'sampled_candidates'.
# 10. Sample candidates are relaxed for 10 steps in local GPR. 
# 11. All 'evaluated_candidates' are added to Database. 





        




