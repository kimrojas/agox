import numpy as np
from scipy.spatial.distance import euclidean

from ase.calculators.calculator import Calculator
from time import time
from ase.calculators.singlepoint import SinglePointCalculator



class krr_calculator(Calculator):

    implemented_properties = ['energy', 'forces']
    default_

    def __init__(self, MLmodel, label='MLmodel', kappa=None, **kwargs):
        self.MLmodel = MLmodel
        self.kappa = kappa
        
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):

        Calculator.calculate(self, atoms, properties, system_changes)

        if 'energy' in properties:
            if self.kappa is None:
                E = self.MLmodel.predict_energy(atoms, return_error=False)
            else:
                energy, error, _ = self.MLmodel.predict_energy(atoms, return_error=True)
                E = energy - self.kappa*error
            self.results['energy'] = E

        if 'forces' in properties:
            if self.kappa is None:
                F = self.MLmodel.predict_force(atoms).reshape((-1,3))
            else:
                F, F_error = self.MLmodel.predict_force(atoms, return_error=True)
                F = (F + self.kappa*F_error).reshape((-1,3))
            self.results['forces'] = F



class doubleLJ_calculator(Calculator):

    implemented_properties = ['energy', 'forces']
    default_

    def __init__(self, eps=1.8, r0=1.1, sigma=np.sqrt(0.02), label='doubleLJ', noZ=False, **kwargs):
        self.eps = eps
        self.r0 = r0
        self.sigma = sigma

        self.noZ = noZ
        
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=['positions']):
        Calculator.calculate(self, atoms, properties, system_changes)

        if 'energy' in properties:
            self.results['energy'] = self.energy(atoms)
        if 'forces' in properties:
            F = self.forces(atoms)
            if self.noZ:
                F[:,-1] = 0
            self.results['forces'] = F
        
            
    def energy(self, a):
        x = a.get_positions()
        E = 0
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                if j > i:
                    r = euclidean(xi, xj)
                    E1 = 1/r**12 - 2/r**6
                    E2 = -self.eps * np.exp(-(r - self.r0)**2 / (2*self.sigma**2))
                    E += E1 + E2
        return E

    def forces(self, a):
        x = a.get_positions()
        Natoms, dim = x.shape
        dE = np.zeros((Natoms, dim))
        for i, xi in enumerate(x):
            for j, xj in enumerate(x):
                r = euclidean(xi,xj)
                if j != i:
                    rijVec = xi-xj

                    dE1 = 12*rijVec*(-1 / r**14 + 1 / r**8)
                    dE2 = self.eps*(r-self.r0)*rijVec / (r*self.sigma**2) * np.exp(-(r - self.r0)**2 / (2*self.sigma**2))
                    
                    dE[i] += dE1 + dE2
        return -dE
