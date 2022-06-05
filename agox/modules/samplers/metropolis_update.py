import numpy as np
from agox.modules.samplers.metropolis import MetropolisSampler

class MetropolisSamplerUpdated(MetropolisSampler):

    def __init__(self, acceptance_target=0.5, consideration_range=10, **kwargs):
        super().__init__(**kwargs)
        self.acceptance_target = acceptance_target
        self.acceptance_list = []
        self.consideration_range = consideration_range
        self.temperature_step = 0.1

    def setup(self):
        accepted = 0
        if self.chosen_candidate is None:
            self.chosen_candidate = self.database.get_most_recent_candidate()
        else:
            potential_step = self.database.get_most_recent_candidate()

            Eold = self.chosen_candidate.get_potential_energy()
            Enew = potential_step.get_potential_energy()
            if Enew < Eold:
                self.chosen_candidate = potential_step
                accepted = 1
            else:
                P = np.exp(-(Enew - Eold)/self.temperature)
                r = np.random.rand()
                if r < P:
                    self.chosen_candidate = potential_step
                    accepted = 1

        self.acceptance_list.append(int(accepted))
        # Other things may rely on the length of the sample.
        if self.chosen_candidate is not None:
            self.sample = [self.chosen_candidate]

        self.update_temperature()

    def update_temperature(self):
        if len(self.acceptance_list) >= self.consideration_range:
            acceptance_percentage = np.mean(self.acceptance_list[-self.consideration_range:])

            if np.abs(acceptance_percentage - self.acceptance_target) < 0.035:
                pass
            elif acceptance_percentage < self.acceptance_target:
                self.temperature += self.temperature_step
            elif acceptance_percentage > self.acceptance_target and self.temperature > 0.1:
                self.temperature -= self.temperature_step
        
            self.writer('Acceptance Percentage: {}'.format(acceptance_percentage))
            self.writer('Temperature {}'.format(self.temperature))