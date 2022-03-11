from .acquisitor_LCB import LowerConfidenceBoundAcquisitor
import numpy as np

class ClusterLowerConfidenceBoundAcquisitor(LowerConfidenceBoundAcquisitor):

    name = 'ClusterLCBAcquisitor'

    def __init__(self, sampler, **kwargs):
        super().__init__(**kwargs)
        self.sampler = sampler
        
    def sort_according_to_acquisition_function(self, candidates):        
        """
        Calculates acquisiton-function based on the implemeneted version calculate_acquisition_function. 

        Note: Sorted so that the candidate with the LOWEST acquisition function value is first. 
        """
        acquisition_values = self.calculate_acquisition_function(candidates)
        sort_idx = np.argsort(acquisition_values)
        sorted_candidates = [candidates[i] for i in sort_idx]
        acquisition_values = acquisition_values[sort_idx]          

        if self.verbose and self.model_calculator.ever_trained and len(self.sampler.sample) > 0:

            for candidate in sorted_candidates[::-1]:
                E = candidate.get_potential_energy()
                sigma = candidate.get_meta_information('uncertainty')
                E0 = candidate.closest_sample.get_meta_information('model_energy')
                sigma0 = candidate.closest_sample.get_meta_information('uncertainty')
                print('FITNESS E={:8.3f} E0={:8.3f} dE={:8.3f} (E-E0)-k*s={:8.3f} s={:8.3f} s0={:8.3f}'.format(E,
                                                                                                               E0,
                                                                                                               E-E0,
                                                                                                               E-E0-self.kappa*sigma,
                                                                                                               sigma,
                                                                                                               sigma0))
        return sorted_candidates, acquisition_values

    def calculate_acquisition_function(self, candidates):
        fitness = np.zeros(len(candidates))
        
        if self.model_calculator.ready_state:
            # Attach calculator and get model_energy
            for i, candidate in enumerate(candidates):
                candidate.set_calculator(self.model_calculator)
                E = candidate.get_potential_energy()
                sigma = candidate.get_uncertainty()

                if self.verbose:
                    # only for printing purposes
                    candidate.add_meta_information('model_energy', E)
                    candidate.add_meta_information('uncertainty',sigma)

                if len(self.sampler.sample) > 0:
                    closest_sample = self.sampler.assign_to_closest_sample(candidate)
                    assert closest_sample is not None,'Sampler did not return a closest sample, which is required for {}'.format(self.__class__)
                    candidate.closest_sample = closest_sample
                    E0 = closest_sample.get_meta_information('model_energy')
                    sigma0 = closest_sample.get_meta_information('uncertainty')
                else:
                    E0 = 0
                    sigma0 = 0

                fitness[i] = (E - E0) - self.kappa * sigma

        return fitness

    def book_keeping(self, state, candidate):
        if hasattr(candidate,'closest_sample'):
            closest_sample = candidate.closest_sample
            delattr(candidate,'closest_sample')
            if state:
                self.sampler.adjust_sample_cluster_distance(candidate,closest_sample)
        super().book_keeping(state,candidate)
