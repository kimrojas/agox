import numpy as np
from agox.modules.postprocessors.postprocess_ABC import PostprocessBaseClass
#from ase.io import write
from scipy.spatial.distance import cdist
from ase.data import covalent_radii

class SortingPostProcess(PostprocessBaseClass):

    name = 'Sorter'

    def __init__(self, c1=0.75, c2=1.25, may_nucleate_at_several_places=False, **kwargs):
        super().__init__(**kwargs)

        self.c1 = c1
        self.c2 = c2
        self.may_nucleate_at_several_places = may_nucleate_at_several_places
        self.debug_i = 0

    def _place_the_rest(self, distances_rel, Ntotal, placed_atoms_indices):
        indices_to_place = [i for i in range(Ntotal) if i not in placed_atoms_indices]
        while indices_to_place:
            nothing_could_be_placed = True
            for i in indices_to_place:
                c = distances_rel[i][placed_atoms_indices]
                if np.any(np.logical_and(self.c1 <= c, c <= self.c2)):
                    placed_atoms_indices.append(i)
                    indices_to_place = [j for j in indices_to_place if j != i] 
                    nothing_could_be_placed = False
                    break
            if nothing_could_be_placed:
                return None
        return placed_atoms_indices

    @PostprocessBaseClass.immunity_decorator
    def postprocess(self, candidate):        
        if candidate is None:
            print('Not buildable (since already None)')
            return None

        Ntemplate = len(candidate.template)
        N = len(candidate)

        distances_abs = candidate.get_all_distances(mic=True)
        r = [covalent_radii[atom.number] for atom in candidate]
        x,y = np.meshgrid(r,r)
        optimal_distances = x+y
        distances_rel = distances_abs / optimal_distances

        any_placed_atom_too_close_to_template = np.any(distances_rel[:Ntemplate,Ntemplate:] < self.c1)
        if any_placed_atom_too_close_to_template:
            print('Not buildable as some atom(s) too close to template')
            return None

        assert self.c1 < 1,'c1 is so large that this trick with np.eye (to avoid self-interaction) does not work'
        any_two_placed_atoms_too_close_to_each_other = np.any(distances_rel[Ntemplate:,Ntemplate:] + np.eye(N-Ntemplate) < self.c1)
        if any_two_placed_atoms_too_close_to_each_other:
            print('Not buildable as some atoms too close to each other')
            return None

        found_an_atom_from_which_the_rest_could_be_built = False

        # if there is a template this loop may provide the buildable version
        for first_atom_index in range(Ntemplate,N):

            # see if this atom may be placed
            c = distances_rel[first_atom_index,:Ntemplate]
            if not np.any(np.logical_and(self.c1 <= c, c <= self.c2)):
                continue

            # see if the remaining atoms may be placed
            if not self.may_nucleate_at_several_places:
                # remove template atoms while placing the rest
                d = distances_rel[Ntemplate:,Ntemplate:]
                Ntotal = N - Ntemplate
                placed_atoms_indices = [first_atom_index - Ntemplate]
                placed_atoms_indices = self._place_the_rest(d, Ntotal, placed_atoms_indices)
                if placed_atoms_indices is not None:
                    placed_atoms_indices = list(range(Ntemplate)) + [i + Ntemplate for i in placed_atoms_indices]
                    found_an_atom_from_which_the_rest_could_be_built = True
                else:
                    found_an_atom_from_which_the_rest_could_be_built = False
                break
            else:
                d = distances_rel
                Ntotal = N
                placed_atoms_indices = list(range(Ntemplate))
                placed_atoms_indices.append(first_atom_index)
                placed_atoms_indices = self._place_the_rest(d, Ntotal, placed_atoms_indices)
                if placed_atoms_indices is not None:
                    found_an_atom_from_which_the_rest_could_be_built = True
                else:
                    found_an_atom_from_which_the_rest_could_be_built = False
                break

        # there is no template the loop was bypassed and we need to check separately
        if Ntemplate == 0:
            d = distances_rel
            Ntotal = N
            placed_atoms_indices = [0]
            placed_atoms_indices = self._place_the_rest(d, Ntotal, placed_atoms_indices)
            if placed_atoms_indices is not None:
                found_an_atom_from_which_the_rest_could_be_built = True
            else:
                found_an_atom_from_which_the_rest_could_be_built = False

        if not found_an_atom_from_which_the_rest_could_be_built:
            print('Not sortable or buildable')
            return None

        description = candidate.get_meta_information('description')
        if placed_atoms_indices != list(range(N)):
            if self.verbose:
                print('Sorted atoms:',description,placed_atoms_indices)
            else:
                print('Sorted atoms:',description,placed_atoms_indices[Ntemplate:])
        buildable_version_of_candidate = candidate.copy()
        for i,j in enumerate(placed_atoms_indices):
            if i >= Ntemplate:
                buildable_version_of_candidate[i].position = candidate[j].position
                buildable_version_of_candidate[i].number = candidate[j].number
        print('Buildable',description)
        return buildable_version_of_candidate

