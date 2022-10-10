import numpy as np
from agox.acquisitors.ABC_acquisitor import AcquisitorBaseClass

class MinimumEnergyAcquisitor(AcquisitorBaseClass):

    name = 'MEAcquisitor'

    def __init__(self, model_calculator, **kwargs):
        super().__init__(**kwargs)
        self.model_calculator = model_calculator

    def calculate_acquisition_function(self, candidates):
        fitness = np.zeros(len(candidates))
        if self.model_calculator.ready_state:
            # Attach calculator and get model_energy
            for i, candidate in enumerate(candidates):
                candidate.set_calculator(self.model_calculator)
                E = candidate.get_potential_energy()
                self.writer(f'ME acquisitor energy: {E:.3f}')

        return fitness

    def assign_from_main(self, main):
        super().assign_from_main(main)
