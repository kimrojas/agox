import numpy as np
from agox.models.ensemble import Ensemble

class EnsemblePerc(Ensemble):

    def __init__(self, data_percentage=0.8, **kwargs):
        super().__init__(**kwargs)
        self.data_percentage = data_percentage

    def train_model(self, training_data, **kwargs):

        self.writer('Training ensemble of models with {}% of the training data'.format(self.data_percentage*100))
        self.writer('Data size: {}'.format(len(training_data)))

        for model in self.models:

            # Get a random subset of the training data
            n_data = len(training_data)
            n_data_subset = int(self.data_percentage * n_data)
            subset_indices = np.random.choice(n_data, n_data_subset, replace=False)
            subset_data = [training_data[i] for i in subset_indices]
            self.writer('Training model with {} data points'.format(len(subset_data)))

            model.train_model(subset_data, **kwargs)
        
        self.ready_state = True
