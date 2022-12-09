
#import numpy as np

import jax.numpy as np
from jax import vmap, grad, jacfwd
from jax.scipy.special import erf
from ase.neighborlist import neighbor_list
from itertools import combinations_with_replacement, combinations

from agox.models.descriptors import DescriptorBaseClass

class Fingerprint(DescriptorBaseClass):
    '''
        jax-implementation of Malthe's modified fingerprint descriptor
        based on agox/models/gaussian_process/featureCalculators_multi/angular_fingerprintFeature_cy.pyx
    '''

    name = 'JaxFingerprint'

    feature_types = ['global', 'global_gradient']

    def __init__(self, r_cutoff_radial=4, r_cutoff_angular=4, smearing_width_radial=0.2, smearing_width_angular=0.2,
                bin_width_radial=0.1, bin_num_angular=30,
                eta=1, gamma=3, **kwargs) -> None:
        '''
            r_cutoff_radial         :   cutoff radius (Å) for radial feature
            r_cutoff_angular        :   cutoff radius (Å) for angular feature
            smearing_width_radial   :   width (Å) of Gaussian smearing for radial feature
            smearing_width_angular  :   width (rad) of Gaussian smearing for angular feature
        '''
        super().__init__(self, **kwargs)

        self.r_cutoff_radial = r_cutoff_radial
        self.r_cutoff_angular = r_cutoff_angular
        self.smearing_width_radial = smearing_width_radial
        self.smearing_width_angular = smearing_width_angular
        self.bin_width_radial = bin_width_radial
        self.bin_num_radial = np.ceil(self.r_cutoff_radial/self.bin_width_radial).astype(np.int8)
        self.bin_num_angular = bin_num_angular


    def load_atoms_info(self, atoms):
        '''
            important to check how many atoms are available for each element,
            and which atom is used as the center atom in a angle
        '''
        n_atoms = len(atoms)
        atomic_numbers = atoms.get_atomic_numbers()
        unique_elements, counts_in_each_element = np.unique(atomic_numbers, return_counts=True)
        n_elements = unique_elements.shape[0]
        all_bonds = list(combinations_with_replacement(unique_elements, 2))
        unique_bonds = []
        for bond in all_bonds:
            for element, count in zip(unique_elements, counts_in_each_element):
                if bond.count(element) > count:
                    break
            else:
                unique_bonds.append(bond)
        unique_bonds = np.array(unique_bonds)

        n_type_radial = len(unique_bonds)
        n_type_angular = len(list(combinations_with_replacement(unique_elements, 3)))
        #n_type_angular = 6
        #all_triplets = list(combinations_with_replacement(unique_elements, 3))
        all_triplets = np.array(list(combinations(atomic_numbers, 3)))
        unique_triplets = np.unique(all_triplets, axis=0)
        
        n_feature_radial = n_type_radial * self.bin_num_radial
        n_feature_angular = n_type_angular * self.bin_num_angular

        #print(n_type_radial, n_feature_radial, self.bin_num_radial)
        #print(n_type_angular, n_feature_angular)
        #for trip in unique_triplets:
        #    print(trip)

        self.n_atoms = n_atoms
        self.volume = atoms.get_volume()
        self.unique_bonds = unique_bonds
        self.n_type_radial = n_type_radial
        self.n_feature_radial = n_feature_radial
        self.n_feature_angular = n_feature_angular

    def convert_atoms(self, atoms, cutoff=4):
        """
        Calculate all bond distances with NeighborList
        """
        idx_i, idx_j = neighbor_list('ij', atoms, cutoff)
        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        bond_types = []
        for i, j in zip(idx_i, idx_j):
            for idx_bond in range(self.n_type_radial):
                if (np.sort(atomic_numbers[[i, j]]) == self.unique_bonds[idx_bond]).all():
                    bond_types.append(idx_bond)                
        bond_types = np.array(bond_types)

        return idx_i, idx_j, positions, bond_types

    def compute_distances(self, positions, idx_i, idx_j):
        R_i = positions[idx_i]
        R_j = positions[idx_j]
        distances = np.linalg.norm(R_i - R_j, axis=1)

        return distances

    def f_angular_cutoff(self, r, r_cutoff_angular, gamma):
        return 1 + gamma * (r/r_cutoff_angular) ** (gamma+1) - (gamma+1)*(r/r_cutoff_angular) ** gamma
    
    def get_radial_feature(self, positions, idx_i, idx_j, bond_types):
        n_atoms = self.n_atoms
        volume = self.volume
        bin_width_radial = self.bin_width_radial
        bin_num_radial = self.bin_num_radial
        smearing_width_radial = self.smearing_width_radial
        r_cutoff_radial = self.r_cutoff_radial
        r_in_bin = np.linspace(0, r_cutoff_radial, bin_num_radial)
        radial_feature = np.zeros(self.n_feature_radial)
        distances = self.compute_distances(positions, idx_i, idx_j)
        
        n_sigmas = 4
        m1 = n_sigmas * smearing_width_radial / bin_width_radial
        smearing_norm1 = 1 / erf(1/np.sqrt(2) * m1 * bin_width_radial/smearing_width_radial)


        for i, j, r_ij, idx_bond in zip(idx_i, idx_j, distances, bond_types): 
            if r_ij < 1e-6 or r_ij >= r_cutoff_radial:
                continue
            
            norm_factor_radial = smearing_norm1 / (4 * np.pi * r_ij**2 * bin_width_radial * n_atoms**2 / volume)
            feature_in_bin = norm_factor_radial * np.exp(-(r_in_bin-r_ij)**2 / (2 * smearing_width_radial**2))
            
            indices_bond = np.arange(idx_bond*bin_num_radial,idx_bond*bin_num_radial+bin_num_radial)
            radial_feature = radial_feature.at[indices_bond].add(feature_in_bin)

        return radial_feature 

    def get_radial_feature_gradient(self, positions, idx_i, idx_j, bond_types):
        grad_fn = jacfwd(self.get_radial_feature, argnums=0)
        radial_gradeint = grad_fn(positions, idx_i, idx_j, bond_types).reshape(-1, self.n_feature_radial)

        #jax_grad_func = vmap(grad(self.get_radial_feature, argnums=0, allow_int=True), (None, 0))
        #df = jax_grad_func(positions, idx_i, idx_j, bond_types)
        #print(df)
        return radial_gradeint

    def create_global_features(self, atoms):
        self.load_atoms_info(atoms)
        idx_i, idx_j, positions, bond_types = self.convert_atoms(atoms)
        radial_feature = self.get_radial_feature(positions, idx_i, idx_j, bond_types)

        return np.asarray(radial_feature)

    def create_global_feature_gradient(self, atoms):
        self.load_atoms_info(atoms)
        idx_i, idx_j, positions, bond_types = self.convert_atoms(atoms)
        radial_gradient = self.get_radial_feature_gradient(positions, idx_i, idx_j, bond_types)

        return np.asarray(radial_gradient)




if __name__ == '__main__':

    from ase.build import molecule
    atoms = molecule('C6H6')
    test = Fingerprint()

    test.load_atoms_info(atoms)