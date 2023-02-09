from ase import Atoms
import functools

def candidate_list_comprehension(func):
    @functools.wraps(func)
    def wrapper(self, possible_list, *args, **kwargs):
        if isinstance(possible_list, list):
            results = []
            for item in possible_list:
                if isinstance(item, Atoms):
                    results.append(func(self, item, *args, **kwargs))
                else:
                    raise TypeError('List must be list of Atoms (or Candidate) objects.')
            return results
        elif isinstance(possible_list, Atoms):
            return func(self, possible_list, *args, **kwargs)
        else:
            raise TypeError('Object must be of type Atoms (or Candidate).')
    return wrapper

