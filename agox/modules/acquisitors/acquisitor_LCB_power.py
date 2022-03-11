import numpy as np
from agox.modules.acquisitors import LowerConfidenceBoundAcquisitor, LowerConfidenceBoundCalculator

class PowerLowerConfidenceBoundAcquisitor(LowerConfidenceBoundAcquisitor):

    name = 'PowerLowerConfindenceBoundAcquisitor'

    def __init__(self, model_calculator, power=0, **kwargs):
        super().__init__(model_calculator, **kwargs)
        self.power = power

    def acquisition_function(self, E, sigma):
        return E - self.kappa * sigma ** self.power

    def acquisition_derivative(self, E, F, sigma, sigma_force):
        return F + self.kappa * self.power * sigma**(self.power-1) * sigma_force

