import numpy as np
from agox.environments.ABC_environment import EnvironmentBaseClass
from ase.atoms import symbols2numbers
from ase.symbols import Symbols

class Environment(EnvironmentBaseClass):

    def __init__(self, template, numbers=None, symbols=None, **kwargs):
        super().__init__(**kwargs)

        # Both numbers and symbols cannot be specified:
        assert (numbers is not None) is not (symbols is not None) # XOR

        if numbers is not None:
            self._numbers = numbers
        elif symbols is not None:
            self._numbers = symbols2numbers(symbols)
        
        self._template = template

    def get_template(self):
        return self._template.copy()

    def set_template(self, template):
        self._template = template

    def set_numbers(self, numbers):
        self._numbers = numbers

    def get_numbers(self):
        return self._numbers.copy()

    def get_missing_types(self):
        return np.sort(np.unique(self.get_numbers()))

    def get_all_types(self):
        return list(set(list(self._template.numbers) + list(self.get_numbers())))

    def get_identifier(self):
        return self.__hash__()

    def get_missing_indices(self):
        return np.arange(len(self._template), len(self._template)+len(self._numbers))

    def get_all_numbers(self):
        all_numbers = np.append(self.get_numbers(), self._template.get_atomic_numbers())
        return all_numbers

    def get_all_species(self):
        return list(Symbols(self.get_all_types()).species())

    def match(self, candidate):
        cand_numbers = candidate.get_atomic_numbers()
        env_numbers = self.get_all_numbers()

        stoi_match = (np.sort(cand_numbers) == np.sort(env_numbers)).all() # Not very efficient, but does it matter? Should pobably only use this function for debugging.
        template_match = (candidate.positions[0:len(candidate.template)] == self._template.positions).all()
        return stoi_match*template_match

    def __hash__(self):
        feature = tuple(self.get_numbers()) + tuple(self._template.get_atomic_numbers()) + tuple(self._template.get_positions().flatten().tolist())
        return hash(feature)    



if __name__ == '__main__':

    from ase import Atoms

    template = Atoms('H1', positions=[[5, 5, 5]], cell=np.eye(3)*10)

    numbers = [1, 1]

    env = EnvironmentSingular(template, numbers=numbers)

    print(env.get_identifier())

    