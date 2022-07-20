import numpy as np
from agox.modules.acquisitors.LCB import LowerConfidenceBoundAcquisitor


class FilteredLowerConfidenceBoundAcquisitor(LowerConfidenceBoundAcquisitor):

    def __init__(self, maximum_uncertainty=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maximum_uncertainty = maximum_uncertainty

    def calculate_acquisition_function(self, candidates):
        base_fitness = super().calculate_acquisition_function(candidates)

        if self.model_calculator.ready_state:
            uncertainties = np.array([candidate.get_meta_information('uncertainty') for candidate in candidates])
            K0 = np.exp(self.model_calculator.model.kernel.get_params()['k1'].get_params()['k1'].theta)

            certainties = uncertainties / np.sqrt(K0)

            min_certainty = self.maximum_uncertainty

            for _ in range(5):
                filtered = certainties < min_certainty

                if filtered.any():               
                    penalty = np.array([1000 if not filtered[i] else 0 for i in range(len(candidates))])
                    fitness = base_fitness + penalty
                    return fitness
                else: 
                    min_certainty = min_certainty + (1-min_certainty)/2
            
        return base_fitness


