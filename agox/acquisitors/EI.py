import numpy as np
from scipy.stats import norm
from agox.acquisitors import AcquisitorBaseClass
from agox.observer import Observer
from agox.writer import agox_writer

class ExpectedImprovementAcquisitor(AcquisitorBaseClass):
    
    name = 'EIAcquisitor'
    
    def __init__(self, model, database, xi=0.01, **kwargs):
        super().__init__(maximize=True, **kwargs)
        self.model = model
        self.xi = xi
        self.lowest_energy = 10E10
        self.add_observer_method(self.update, sets={}, gets={}, order=0, handler_identifier='database')
        self.attach(database)

    def calculate_acquisition_function(self, candidates):
        fitness = np.zeros(len(candidates))
        for i, candidate in enumerate(candidates):
            E, sigma = self.model.predict_energy(candidate, return_uncertainty=True)
            fitness[i] = self.acquisition_function(E, sigma)
            candidate.add_meta_information('model_energy', E)
            candidate.add_meta_information('uncertainty', sigma)
        return fitness
    
    def acquisition_function(self, E, sigma):
        mu = E
        mu_opt = self.lowest_energy
        with np.errstate(divide='warn'):
            imp = mu_opt - mu - self.xi
            Z = imp / sigma 
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei
    
    def print_information(self, candidates, acquisition_values):
        if self.do_check():
            for i, candidate in enumerate(candidates):
                fitness = acquisition_values[i]
                E = candidate.get_meta_information('model_energy')
                sigma = candidate.get_meta_information('uncertainty')
                self.writer(f'Candidate {i}: E = {E:8.3f}, s = {sigma:8.3f}, EI = {fitness:8.6f}')

    @agox_writer
    @Observer.observer_method
    def update(self, database, state):
        self.lowest_energy = database.get_best_energy()
        self.writer('Lowest energy: {}'.format(self.lowest_energy))

    def do_check(self, **kwargs):
        return self.model.ready_state

