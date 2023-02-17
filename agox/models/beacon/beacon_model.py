import numpy as np
from agox.models.ABC_model import ModelBaseClass
from ase.calculators.calculator import all_changes, Calculator
#from gpatom.gpfp. import HyperparameterFitter, PriorDistributionLogNormal


class BeaconModel(ModelBaseClass):

   
    implemented_properties = ['energy', 'forces', 'uncertainty', 'force_uncertainty']

    name = 'BeaconGPR'
        
    dynamic_attributes = ['gp']     
   
    def __init__(self, gp=None, fp=None, hp_optimizer=None):  
        super().__init__()
        self.gp=gp
        self.descriptor=fp
        self.hp_optimizer=hp_optimizer

    
    def new_fingerprint(self, atoms): #, *args, **kwargs):
        '''
        Compute a fingerprint of 'atoms' with given parameters.
        '''
        return self.descriptor.get(atoms=atoms) #, *args, **kwargs)


    def train_model(self, training_data, **kwargs):
        '''
        Train a Gaussian process with given data.

        Parameters
        ----------
        gp : GaussianProcess instance
        data : Database instance

        Return a trained GaussianProcess instance
        '''
        features, targets = self.get_data_properties(training_data)
        
        
        
        if self.hp_optimizer is not None:,
            if len(training_data)>2:
            
                self.hp_optimizer.fit(self.gp)    
            
            pd=PriorDistributionLogNormal(loc=7.5, width=0.05)
            HyperparameterFitter.fit(self.gp, ['scale'], fit_weight=True, fit_prior=True, pd=pd)
            # yes, this works!!!  no negative uncertainties!
            # CL: note that the hp_fitter this way easily could
            # be imported from the outside and hence be set to whatever.
            
        self.gp.train(features, targets)
        
        self.ready_state = True

        
    def get_data_properties(self, data):

        X=[]
        Y=[]
 
        for d in data:
            d.copy()
         
            X.append(self.descriptor.get_fp_object(d))     
         
            Y.append(d.get_potential_energy())
         
            if self.gp.use_forces:
                forces= -1 * d.get_forces().flatten()
                Y+=list(forces)
             
        X=np.array(X)
        Y=np.array(Y)

        return X, Y



    def get_properties(self, atoms):
        
        x=self.descriptor.get_fp_object(atoms)
        
        f, V = self.gp.predict(x, get_variance=True)
                
        print('AM I NEGATIVE')
             
        energy=f[0]
        forces=-1*f[1:].reshape(-1, 3)
        
        if self.gp.use_forces:
            
            print(V[0,0])
            uncertainty_energy = np.sqrt(V[0, 0])
            uncertainty_forces= np.sqrt(   np.diag(V[1:,1:])  ).reshape(-1, 3)
        else:
            
            print(V[0])
            uncertainty_energy=np.sqrt(V[0])
            uncertainty_forces=np.zeros_like(forces)
        
        return energy, forces, uncertainty_energy, uncertainty_forces




    def predict_energy(self, atoms, return_uncertainty=False, **kwargs):
      #  x=self.descriptor.get_fp_object(atoms)
        energy, forces, unc_energy, unc_forces = self.get_properties(atoms)
        
        if return_uncertainty:
            return energy, unc_energy
        return energy


    def predict_forces(self, atoms, return_uncertainty=False, **kwargs):
      #  x=self.descriptor.get_fp_object(atoms)
        energy, forces, und_energy, unc_forces = self.get_properties(atoms)
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
       Calculator.calculate(self, atoms, properties, system_changes)

       E, sigma = self.predict_energy(atoms, return_uncertainty=True)
       self.results['energy'] = E
       
       if 'forces' in properties:
           forces = self.predict_forces(atoms)
           self.results['forces'] = forces
       
       self.results['uncertainty'] = sigma
       #self.results['force_uncertainties'] = 0




from agox.writer import Writer, agox_writer
from agox.observer import Observer

class HPObserver(Observer, Writer):

    name = 'BeaconHP_¨Observer'

    def __init__(self, order=3, gets={'get_key':'hyper parameters'}, model=None):
        Observer.__init__(self, gets=gets, order=order)
        Writer.__init__(self)
        self.model=model
        self.add_observer_method(self.basic_observer_method, order=self.order[0],
            sets=self.sets[0], gets=self.gets[0])

    @agox_writer
    @Observer.observer_method
    def basic_observer_method(self, state):
        self.writer(f'Iteration: {self.get_iteration_counter()}')

        hp=self.model.gp.hp
        prior=self.model.gp.prior.constant
          
        self.writer(f'hyperparameters {hp}') 
        self.writer(f'priorconstant {prior}')  
        
        
        
        
        
        
        
'''        

so tomorrow, do tasks in this order
1. get the fp.caching system.    it wasnt ready yet... 
2. figure out how to make a branch for ghost in Agox
3. fix ICE
4. clean up
           

to fix ICE
make a ICE_beacon_model script.
inherit from beacon_model.
overwrite relevant functions. 


agox interface should be structured in such a way, that all modules from beacon can just be directly
inserted into agox.  in that way fixing something in BEACON scripts will autimatically carry over to Agox. 
then I dont have to do double work.

'''        

