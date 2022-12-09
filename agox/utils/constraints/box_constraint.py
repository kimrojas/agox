from types import new_class
import numpy as np

class BoxConstraint:

    def __init__(self, confinement_cell=None, confinement_corner=None, indices=None, **kwargs):
        """
        Constraint that wont allow atoms to move outside of a 'box' or cell defined by the matrix B. 
        Atoms influenced by the constraint are to be specified by indices
        c defines the lower-left corner of the cell/box.

        So if atoms are to be constrained to a unit cell then B = atoms.cell and c = [0, 0, 0]
        """
        self.confinement_cell = np.array(confinement_cell)
        self.confinement_corner = np.array(confinement_corner)
        if indices is None:
            indices = np.array([])
        self.indices = np.array(indices).flatten()

        # Soft boundary & force decay.
        self.lower_soft_boundary = 0.05; self.lower_hard_boundary = 0.001
        self.al, self.bl = np.polyfit([self.lower_soft_boundary, self.lower_hard_boundary], [1, 0], 1)
        self.upper_soft_boundary = 0.95; self.upper_hard_boundary = 0.999
        self.au, self.bu = np.polyfit([self.upper_soft_boundary, self.upper_hard_boundary], [1, 0], 1)

        if np.all(self.confinement_cell[:, 2] == 0):
            self.dimensionality = 2
            self.effective_confinement_cell = self.confinement_cell[0:2, 0:2]
            if np.all(self.confinement_cell[:, 1] == 0):
                self.effective_confinement_cell = self.confinement_cell[0:1, 0:1]
                self.dimensionality = 1
        else:
            self.effective_confinement_cell = self.confinement_cell
            self.dimensionality = 3
    
    def linear_boundary(self, x, a, b):
        return a*x+b

    def adjust_positions(self, atoms, newpositions):
        inside = self.check_if_inside_box(newpositions[self.indices])
        #newpositions[not inside, :] = atoms.positions[not inside]        
        # New positions of those atoms that are not inside (so outside) the box are set inside the box. 
        newpositions[self.indices[np.invert(inside)], :] = atoms.positions[self.indices[np.invert(inside)], :] 

    def adjust_forces(self, atoms, forces):
        C = self.get_projection_coefficients(atoms.positions[self.indices])        
        # Because adjust positions does not allow the atoms to escape the box we know that all atoms are witihn the box. 
        # Want to set the forces to zero if atoms are close to the box, this happens if any component of C is close to 0 or 1. 
        for coeff, idx in zip(C, self.indices):
            if ((coeff < 0) * (coeff > 1)).any():
                forces[idx] = 0 # Somehow the atom is outside, so it is just locked. 
            if (coeff > self.upper_soft_boundary).any():
                forces[idx] = self.linear_boundary(np.max(coeff), self.au, self.bu) * forces[idx]
            elif (coeff < self.lower_soft_boundary).any():
                forces[idx] = self.linear_boundary(np.min(coeff), self.al, self.bl) * forces[idx]

    def get_removed_dof(self, atoms):
        return 3*len(atoms)
        
    def get_projection_coefficients_old(self, positions):
        return np.linalg.solve(self.confinement_cell, (positions-self.confinement_corner).T).T

    def get_projection_coefficients(self, positions):
        positions = positions.reshape(-1, 3)
        return np.linalg.solve(self.effective_confinement_cell.T, (positions-self.confinement_corner)[:, 0:self.dimensionality].T).T.reshape(-1, self.dimensionality)

    def check_if_inside_box(self, positions):
        """
        Finds the fractional coordinates of the atomic positions in terms of the box defined by the constraint. 
        """        
        C = self.get_projection_coefficients(positions)
        inside = np.all((C > 0) * (C < 1), axis=1)
        return inside

    def get_confinement_limits(self):
        """
        This returns the confinement-limit lists which always assumes a square box. 
        """
        conf = [self.confinement_corner, self.confinement_cell @ np.array([1, 1, 1]) + self.confinement_corner]
        return conf

    def todict(self):
        return {'name':'BoxConstraint', 
                'kwargs':{'confinement_cell':self.confinement_cell.tolist(), 'confinement_corner':self.confinement_corner.tolist(), 'indices':self.indices.tolist()}}
        


# To work with ASE read/write we need to do some jank shit. 
from ase import constraints 
constraints.__all__.append('BoxConstraint')
constraints.BoxConstraint = BoxConstraint

if __name__ == '__main__':
    from ase import Atoms

    B = np.eye(3) * 1
    c = np.array([0, 0, 0])

    atoms = Atoms('H4', positions=[[0.5, 0.5, 0.5], [0.1, 0.1, 0.9], [0.1, 0.1, 0.9], [0.02, 0.5, 0.1]])

    BC = BoxConstraint(B, c, indices=np.array([0, 1, 2, 3]))

    print(BC.check_if_inside_box(atoms.positions))

    forces = np.tile(np.array([[1, 0, 0]]), 4).reshape(4, 3).astype(float)

    print(forces)    
    BC.adjust_forces(atoms, forces)
    print(forces)

    #print(BC.upper_boundary_factor(0.999))
    




        
