import numpy as np
from ase.data import covalent_radii
from abc import ABC, abstractmethod

from agox.modules.candidates import StandardCandidate

dimensionality_angles = {
                        3:{'theta':[0, 2*np.pi], 'phi':[0, np.pi]},
                        2:{'theta':[0, 2*np.pi], 'phi':[np.pi/2, np.pi/2]},
                        1:{'theta':[0, 0], 'phi':[np.pi/2, np.pi/2]}
                        }

class GeneratorBaseClass(ABC):

    def __init__(self, confinement_cell=None, cell_corner=None, c1=0.75, c2=1.25, dimensionality=3):
        """
        B: Matrix that defines that defines the cell atoms can be placed in. 
        c: Lower left corner of cell
        """
        self.confinement_cell = confinement_cell
        self.cell_corner = cell_corner
        self.confined = confinement_cell is not None and cell_corner is not None
        self.c1 = c1 # Lower limit on bond lengths. 
        self.c2 = c2 # Upper limit on bond lengths. 
        self.dimensionality = dimensionality

        self.candidate_instantiator = StandardCandidate
        if self.confined:
            # Checks if the confinement cell matches the given dimensionality, this assumes 
            # that a 2D search happens in the XY plane and a 1D search happens on the X axis.
            # Thus the z-column of confinement_cell needs to be zero for 2D and z/y columns for 1D. 
            # Ensures that default methods will generate vectors that obey the dimensionality. 
            if dimensionality == 3:
                assert np.all(np.any(self.confinement_cell > 0, axis=1)), 'Confinement cell does not match dimensionality (Not 3D)'
                self.effective_confinement_cell = self.confinement_cell
            if dimensionality == 2:
                assert np.all(self.confinement_cell[:, 2] == 0), 'Confinemnt cell does not match dimensionality (Not 2D)'
                self.effective_confinement_cell = self.confinement_cell[0:2, 0:2]
            if dimensionality == 1:
                assert np.all(self.confinement_cell[:, 2] == 0) and np.all(self.confinement_cell[:, 1] == 0), 'Confinemnt cell does not match dimensionality (Not 1D)'
                self.effective_confinement_cell = self.confinement_cell[0:1, 0:1]

    @abstractmethod
    def get_candidates(self, sampler, environment):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    def __call__(self, sampler, environment):
        return self.get_candidates(sampler, environment)
    
    ####################################################################################################################
    # Convenience methods:
    ####################################################################################################################

    def convert_to_candidate_object(self, atoms_type_object, template):
        candidate =  self.candidate_instantiator(template=template, positions=atoms_type_object.positions, numbers=atoms_type_object.numbers, 
                                          cell=atoms_type_object.cell)

        candidate.add_meta_information('generator', self.name)

        return candidate

    def check_confinement(self, positions):
        """
        Checks that all of the given positions are within the confinement cell. 
        If B and c have not been specified it always returns True. 
        """
        if self.confined:             
            return self.check_positions_within_confinement(positions).all()
        else:
            return True

    def check_positions_within_confinement(self, positions):
        """
        Returns boolean array indicating which atoms are within the confinement limits. 
        """
        if self.confined:
            positions = positions.reshape(-1, 3)
            C = np.linalg.solve(self.effective_confinement_cell.T, (positions-self.cell_corner)[:, 0:self.dimensionality].T).T.reshape(-1, self.dimensionality)
            return np.all((C > 0) * (C < 1), axis=1)
        else:
            return np.ones(positions.shape[0]).astype(bool)

    def check_new_position(self, candidate, new_position, number, skipped_indices=[]):
        """
        Checks if new positions is not too close or too far to any other atom. 

        Probably not be the fastest implementation, so may be worth it to optimize at some point. 
        """
        state = True
        succesful = False
        for i in range(len(candidate)):
            if i in skipped_indices:
                continue

            covalent_dist_ij = covalent_radii[candidate[i].number] + covalent_radii[number]
            rmin = self.c1 * covalent_dist_ij
            rmax = self.c2 * covalent_dist_ij

            distance = np.linalg.norm(new_position-candidate.positions[i])
            if distance < rmin: # If a position is too close we should just give up. 
                return False
            elif not distance > rmax: # If at least a single position is not too far we have a bond.
                succesful = True
        return succesful * state

    def get_sphere_vector(self, atomic_number_i, atomic_number_j):
        """
        Get a random vector on the sphere of appropriate radii. 

        Behaviour changes based on self.dimensionality: 
        3: Vector on sphere. 
        2: Vector on circle (in xy)
        1: Vector on line (x)
        """
        covalent_dist_ij = covalent_radii[atomic_number_i] + covalent_radii[atomic_number_j]
        rmin = self.c1 * covalent_dist_ij
        rmax = self.c2 * covalent_dist_ij
        r = np.random.uniform(rmin**self.dimensionality, rmax**self.dimensionality)**(1/self.dimensionality)
        return self.get_displacement_vector(r)

    def get_displacement_vector(self, radius):
        theta = np.random.uniform(*dimensionality_angles[self.dimensionality]['theta'])
        phi = np.random.uniform(*dimensionality_angles[self.dimensionality]['phi'])
        displacement = radius * np.array([np.cos(theta)*np.sin(phi),
                                          np.sin(theta)*np.sin(phi),
                                          np.cos(phi)])
        return displacement

    def get_box_vector(self):
        return self.confinement_cell @ np.random.rand(3) + self.cell_corner

    def assign_from_main(self, main):
        self.candidate_instantiator = main.candidate_instantiator

    ####################################################################################################################
    # Convenience methods:
    ####################################################################################################################

    def set_confinement_cell(self, cell, cell_corner):        
        self.confinement_cell = cell
        self.cell_corner = cell_corner
        self.confined = True        

    def set_dimensionality(self, dimensionality):
        self.dimensionality = dimensionality