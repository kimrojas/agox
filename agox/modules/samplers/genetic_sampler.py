import numpy as np
from agox.modules.samplers import SamplerBaseClass
from ase.io import read, write

class GeneticSampler(SamplerBaseClass):

    name = 'GeneticSampler'

    def __init__(self, database=None, comparator=None, population_size=10, **kwargs):        
        super().__init__(**kwargs)
        assert database is not None
        self.database = database

        assert comparator is not None
        self.comparator = comparator

        self.population = []
        self.population_size = population_size
        self.idx = 0

    def get_random_member(self):

        P = self.get_selection_probabilities()
        idx = np.random.choice(np.arange(len(self.population)), 1, p=P)[0]

        member = self.population[idx]
        self.increase_times_used(member)
        return member.copy()

    def increase_times_used(self, member):
        count = member.get_meta_information('used_count')
        if count is None:
            count = 0
        else:
            count += 1
        member.add_meta_information('used_count', count)

    def count_similar(self, member):

        candidates = self.database.get_all_candidates()

        count = 0
        for candidate in candidates:

            if self.comparator(member, candidate):
                count += 1
            
        member.add_meta_information('similar_count', count)

    def setup(self):        
        # If the database has enough members: 
        print(len(self.database))
        if not len(self.database) >= self.population_size:
            return 

        # Get the population_size most recent candidates:
        possible_candidates = self.database.get_recent_candidates(self.population_size)

        # If the population is empty we just take all of them for now: 
        if len(self.population) == 0:
            self.population = possible_candidates
            self.sort_population()
            return 

        # Now we start deciding whether to replace candidates in the population: 
        for candidate in possible_candidates:
            state = self.consider_candidate(candidate)
            if state:
                self.sort_population()
        
        # Compare population to database:
        for member in self.population:
            self.count_similar(member)

        if self.get_episode_counter() % 5 == 0:
            write('population_{}.traj'.format(self.get_episode_counter()), self.population)

        self.print_information()

    def consider_candidate(self, candidate):        
        fitness = self.get_fitness(candidate)

        worst_fitness = self.get_fitness(self.population[-1])

        if fitness < worst_fitness and len(self.population) == self.population_size:
            return False

        for i, member in enumerate(self.population):
            if self.comparator(candidate, member):
                if fitness > self.get_fitness(member):

                    used_count = member.get_meta_information('used_count')
                    
                    del self.population[i]
                    candidate.add_meta_information('used_count', used_count) # Replaced this member so inherit its use count. 
                    self.population.append(candidate)
                    return True
                # If it looks like another member we return regardless of whether it replaces that member. 
                return False

        # If it doesn't look like anything we just replace the worst member of the population:
        del self.population[-1]
        self.population.append(candidate)
    
        return True

    def get_fitness(self, candidate):
        population_energies = [candidate.get_potential_energy() for candidate in self.population]
        e_min = np.min(population_energies)
        e_max = np.max(population_energies)

        p = (candidate.get_potential_energy() - e_min) / (e_max - e_min)

        F = 0.5 * (1 - np.tanh(2*p - 1))

        return F
    
    def sort_population(self):
        if len(self.population):
            fitness = [self.get_fitness(candidate) for candidate in self.population]
            sort_idx = np.argsort(fitness)
            self.population = [self.population[i] for i in sort_idx][::-1]


    def get_selection_probabilities(self):
        N = np.array([member.get_meta_information('used_count') for member in self.population])
        N[N == None] = 0
        N = N.astype(int)
        M = np.array([member.get_meta_information('similar_count') for member in self.population])
        M[M == None] = 0
        M = M.astype(int)
        F = np.array([self.get_fitness(member) for member in self.population])
        U = 1 / np.sqrt(M+1) * 1 / np.sqrt(N + 1)
        P = F * U 
        P = P / np.sum(P)
        return P 

    def print_information(self):
        probs = self.get_selection_probabilities()
        print('='*50)
        for i, member in enumerate(self.population):
            E = member.get_potential_energy()
            F = self.get_fitness(member)
            P = probs[i]
            print('Member {}: E = {:6.4f}, F = {:6.4f}, P = {:4.2f}'.format(i, E, F, P))
        print('='*50)

class DistanceComparator:

    def __init__(self, descriptor, threshold):
        self.descriptor = descriptor
        self.threshold = threshold

    def __call__(self, candidate_A, candidate_B):
        return self.compare_candidates(candidate_A, candidate_B)

    def compare_candidates(self, candidate_A, candidate_B):
        feature_A = self.get_feature(candidate_A)
        feature_B = self.get_feature(candidate_B)

        return np.linalg.norm(feature_A-feature_B) < self.threshold

    def get_feature(self, candidate):
        feature = candidate.get_meta_information('population_feature')

        if feature is None:
            feature = self.descriptor.get_feature(candidate)
            candidate.add_meta_information('population_feature', feature)
        
        return feature


