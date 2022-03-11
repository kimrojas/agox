import numpy as np

class AcceptanceObserver:

    name = 'AcceptanceObserver'

    def __init__(self, agox, solutions):
        self.collector = agox.collector
        self.acquisitor = agox.acquisitor

        self.solutions = [self.convert_atoms_to_candidates(agox, solution) for solution in solutions]
        self.recorded_acceptance = []

    def observer_function(self):                
        # Fitness of generated candidates:
        fitness = self.collector.values    
        solution_fitness = self.acquisitor.calculate_acquisition_function(self.solutions)
        accepted = solution_fitness < np.min(fitness)
        self.recorded_acceptance.append(accepted)

    def save(self):
        arr = np.array(self.recorded_acceptance)
        np.save('acceptance.npy', arr)

    def convert_atoms_to_candidates(self, agox, atoms_type_object):
        candidate = agox.candidate_instantiator(template=agox.environment.get_template(), positions=atoms_type_object.positions, numbers=atoms_type_object.numbers, 
                                          cell=atoms_type_object.cell)
        return candidate





