import numpy as np
#from agox.models.ABC_model import ModelBaseClass
from ase.calculators.calculator import all_changes, Calculator
from agox.models.beacon.beacon_fit_hp import HyperparameterFitter, PriorDistributionLogNormal
from agox.models.beacon.beacon_model import BeaconModel

class ICEBeacon_Model(BeaconModel):

   
    implemented_properties = ['energy', 'forces', 'uncertainty', 'force_uncertainty']

    name = 'ICEBeaconGPR'
        
    dynamic_attributes = ['gp']     
   
    def __init__(self, gp=None, fp=None):  
        super().__init__()
        self.gp=gp
        self.descriptor=fp


    def predict_energy(self, atoms, return_uncertainty=False, **kwargs):
        
        energy, forces, unc_energy, unc_forces, frac_grads = self.get_properties(atoms)

        if return_uncertainty:
            return energy, unc_energy
        return energy


    def predict_forces(self, atoms, return_uncertainty=False, **kwargs):
        energy, forces, und_energy, unc_forces, frac_grads = self.get_properties(atoms)
        
        forces=np.hstack( (forces, -1*frac_grads.reshape(16,1) ) )
        
        unc_forces=np.hstack(  ( unc_forces, np.zeros_like(frac_grads.reshape(16,1)) )   )
    
        if return_uncertainty:
            return forces, unc_forces
        return forces    


    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
       """ASE Calculator calculate method
       
       Parameters
       ----------
       atoms : ASE Atoms obj or AGOX Candidate object
           The atoms object for to predict properties of.
       properties : :obj: `list` of :obj: `str`
           List of properties to calculate for the atoms
       system_changes : ASE system_changes
           Which ASE system_changes to check for before calculation
       
       Returns
       ----------
       None
       """                   
       
       
       #print(atoms)
       
       Calculator.calculate(self, atoms, properties, system_changes)

#       self.atoms.fractions=atoms.fractions        
#       self.atoms.n_ghost=atoms.n_ghost 

        # doesnt work 

       E, sigma = self.predict_energy(atoms, return_uncertainty=True)
       #E, sigma = self.predict_energy(self.atoms, return_uncertainty=True)       # is it necessary that it is self.atoms???
       self.results['energy'] = E
       
       if 'forces' in properties:
      #     forces = self.predict_forces(self.atoms)
           forces = self.predict_forces(atoms)
           self.results['forces'] = forces
       
       self.results['uncertainty'] = sigma
      # self.results['force_uncertainties'] = 0

