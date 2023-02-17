#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:02:53 2022

@author: casper
"""


#from gpatom.beacon.icebeacon.gp.gp import ICEGaussianProcess
#from gpatom.beacon.icebeacon.gp.fingerprint import PartialRadAngFP

#from gpatom.beacon.bacon import BEACON, InitatomsGenerator
from gpatom.beacon.inout import FileWriter

#from gpatom.gpfp.prior import ConstantPrior


from ase.io import write

from ase.calculators.singlepoint import SinglePointCalculator

from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, OptimizeWarning
import numpy as np

#from gpatom.beacon.str_gen import RandomDecoration
import warnings

from ase.constraints import FixAtoms

from gpatom.beacon.icebeacon.drs import drs
from gpatom.beacon.icebeacon.ghosthandler import GhostHandler


from agox.postprocessors.ABC_postprocess import PostprocessBaseClass 

#from agox.utils.ray_utils import RayBaseClass, ray_kwarg_keys

from agox.utils.ray_utils import RayPoolUser

























class RandomFractionGenerator:
    '''
    Generates random fractions with given generating type and constraints.
    '''
        
    def __init__(self, frac_cindex=[], Ghost_mode=False,
                 randomtype='drs',
                 fractional_elements=None, n_real=None, frac_lims=None):        
        #self.atoms = atoms
        #self.constr = constr  
        self.frac_cindex=frac_cindex
        #self.natoms = len(self.atoms)  
        self.randomtype = randomtype
        #self.rng = rng
        self.fractional_elements=fractional_elements
        #self.elements=list(elements),
        self.n_real=n_real
        self.frac_lims=frac_lims
        self.Ghost_mode=Ghost_mode
        
        
        if Ghost_mode and (n_real is None): 
            raise RuntimeError('n_real must be set when running Ghost_mode') 



# super confusing. need to be easy. 
    


    def setup_fractions(self, atoms):
        
        if self.Ghost_mode:
            n_ghost=self.get_n_ghost(atoms)
            fractions, frac_cindex = self.get_fractions_for_ghosts(atoms, n_ghost)            
            
        else:
            n_ghost=0
            fractions, frac_cindex = self.get_fractions_for_ICE(atoms)
            
            
        return fractions, frac_cindex, n_ghost




    def get_fractions_for_ghosts(self, atoms, n_ghost):
            
        if n_ghost<0:
            raise RuntimeError('negative number of ghosts in structure') 
        
        if n_ghost==0:
            fractions=FractionConverter.atoms2fractions(atoms, self.fractional_elements)  
            frac_cindex=np.arange(len(atoms))
        else: 
            fractions=self.get(atoms)
            frac_cindex=self.frac_cindex
         
        return fractions, frac_cindex


         
            
         
    def get_fractions_for_ICE(self, atoms):     
        
        fractions = self.get(atoms)
        frac_cindex = self.frac_cindex
            
        return fractions, frac_cindex




    def get_n_ghost(self, atoms):
        if self.Ghost_mode:
            symbol_list = list(atoms.symbols[:])
            n_0 = symbol_list.count(self.fractional_elements[0])
            n_ghost = n_0 - self.n_real
        else: 
            n_ghost=0
            
        return n_ghost
    
    

    def get(self, atoms):
        
        # NB here it takes an old atoms object and refers to that. could be solved by simply importing atoms when doing get.
        # a new sgen will always be made first anyways. you can also set self.natoms here.  then.  
        
    
        #if self.Ghost_mode:
        #    symbol_list = list(atoms.symbols[:])
        #    n_0 = symbol_list.count(self.fractional_elements[0])  # count of first element
        
            #if n_0==self.n_real:
                #    fractions = FractionConverter.atoms2fractions(atoms, self.fractional_elements)
                #    tags=np.ones(len(atoms), dtype=int)
        
        
        #############
#        tags=np.zeros(len(atoms), dtype=int)
#        for idx in self.frac_cindex:
#            tags[idx]=1
#            atoms.set_tags(tags)
        #############
        
        
        
            
        
        
        if self.randomtype == 'uniform':
            f = self.get_uniform(atoms)
    
        elif self.randomtype=='drs':
            f=self.get_dirichlet_rescale(atoms)
            
        elif self.randomtype=='whole_atoms':
            f=self.get_whole_atoms(atoms)
                    
        else:
            raise RuntimeError('randomtype={:s} not known.'
                               .format(self.randomtype))
        return f
 
    
    def get_uniform(self, atoms, n_ghost=0):
        
        # if the atoms are permuted, the constrain indices are fucked as well.
        # I would like a solution which is insensitive to permutations so there is no chance of fuck ups
        # hmm no.  instead of protecting against permutations, lets just make sure they never happen. its too hard to guard against
        # i.e. the random decoration sgen should just be deleted.  probably the easiest anyways  
        # if we run a best structure from ghost, it should be able to run just those atoms all set to 1 and constrained
        # then the entire 'original atoms' problem dissapears. 
        # 
        
        fmask, n_0 = FractionConverter.set_fractional_atoms(atoms, self.fractional_elements)
        natoms = len(atoms) 
        
        n_ghost=self.get_n_ghost(atoms)

        
        f=np.zeros(natoms)
        ci=self.frac_cindex     
        nci=[i for i in np.arange(natoms) if ( (i not in ci) and fmask[i])]   
        
        
        if len(ci)>0:          
            constrained_fractions=FractionConverter.atoms2fractions(atoms,self.fractional_elements)[ci]
            sum_constrained=sum(constrained_fractions)           
            n_0_remain=n_0- sum_constrained - n_ghost       +  n_ghost*self.frac_lims[0]
            f[ci]=constrained_fractions
        else:
            n_0_remain=n_0 - n_ghost         +  n_ghost*self.frac_lims[0]
        
        
        reverse_fmask=[not b for b in fmask]
        val=n_0_remain/(natoms - len(self.frac_cindex) -  sum(np.array(reverse_fmask, dtype=int)))
        f[nci]=val       
        
        return f


    def get_dirichlet_rescale(self, atoms, n_ghost=0):
        
        '''
        atoms are assigned uniformly distributed random values between 0 and 1 by the drs (dirichlet rescale) algorithm
        drs(nval,vsum,upper_lim). a total of nval random numbers are made. each element is between 0 and upper_lim
        with the sum of the full random vector equalling vsum
        '''
        
        fmask, n_0 = FractionConverter.set_fractional_atoms(atoms, self.fractional_elements)
        natoms = len(atoms)
               
        
        n_ghost=self.get_n_ghost(atoms)
        
        f_sum=n_0-n_ghost
        
        lower_limits=np.ones(natoms)*self.frac_lims[0]        
        upper_limits=np.ones(natoms)*self.frac_lims[1]    #  no need for upper limits
        
        reverse_fmask=[not b for b in fmask]
        nfidx=[i for i in np.arange(natoms) if reverse_fmask[i]] 
        upper_limits[nfidx]=0.0
        lower_limits[nfidx]=0.0
    
        if len(self.frac_cindex)>0:
            ci=self.frac_cindex
            constrained_fractions=FractionConverter.atoms2fractions(atoms,self.fractional_elements)[ci]
            lower_limits[ci]=constrained_fractions
            upper_limits[ci]=constrained_fractions


        f=drs(natoms, f_sum, upper_limits, lower_limits)
        f=np.array(f)
        
        return f
    
    
    def get_whole_atoms(self, atoms):
        f=FractionConverter.atoms2fractions(atoms,self.fractional_elements)
        
        return f

class FractionConverter:   # all Atoms

    @staticmethod
    def set_fractional_atoms(atoms, fractional_elements):   # CL made by me.
        '''
        Takes atoms object and list of fractionalized elements
        to define a vector of which atoms are fractionalized
        and how many of the first fractionalised species exists
        '''
        symbol_list = list(atoms.symbols[:])
        fmask = [(symbol in fractional_elements) for symbol in symbol_list]
        n_0 = symbol_list.count(fractional_elements[0])  # count of first element
                
        return fmask, n_0   
  
    @staticmethod
    def atoms2fractions(atoms, fractional_elements):  # used multiple placess
        '''
        Convert ase.Atoms object to fractions (or integers because
        everything is full atoms)
        '''
        fractions = [(1.0 if atom.symbol == fractional_elements[0] else 0.0)
                     for atom in atoms]
        
        return np.array(fractions)

#    @staticmethod
#    def fractions2integers(fractions, fmask, n_0):
#        '''
#        Convert fractions to integers by proper rounding.
#        '''
        
        # This method is so damn unintuitive to read and understand. 
        # Why doesnt it just take the n_0 highest (or whatever order and set them to 1) 
        # instead of all this. its super confusing
        # then we could idiot proof he procedure by initializing integers as zeros(len(atoms))
        # take the constrained ones, check if they were above or below 0.5 to assign 1 or 0.   
        # assign them to the list.  take the remainder and do the below procedure
        
#        integers = fractions.copy()
#        argsort = np.argsort(fractions[fmask])[::-1]        
#        summ = 0.0
        
        
        # I think this is just a very 
        
#        for i, index in enumerate(argsort):
#            if fmask[index]:
#                integers[index] = (1.0 if summ < n_0 else 0.0)     # set to type 1  if n_0 not reached
#                if integers[index] > 0.5:                          # now the integer is one if not n_0 reached and we update sum  
#                    summ += 1.0
#            else:
#                integers[index] = 0.0      # if not fmask.  always zero. 
                
#        assert sum(integers[fmask]) == n_0

#        return integers

#    @staticmethod
#    def fractions2atoms(fractions, atoms, fractional_elements):     # only used to round fractions and for the writer.  3 locations
       # '''
       # Convert fractions to ase.Atoms object by proper rounding and
       # given fractional_elements.
       # '''
        
#        atoms = atoms.copy()
        
#        fmask, n_0 = FractionConverter.set_fractional_atoms(atoms, fractional_elements)
                      
#        integers = FractionConverter.fractions2integers(fractions, fmask, n_0)
#        indices = [(1 if integers[i] < 0.5 else 0)    
#                   for i in range(len(atoms))]            # just flipping.  a bit confusiing. 
        
        
#        for i, el_index in enumerate(indices):
#            if fmask[i]:
#                atoms.symbols[i] = fractional_elements[el_index]    # just assigning elements if fmask is True

#        assert list(atoms.symbols).count(fractional_elements[0]) == n_0


#        return atoms




    @staticmethod
    def fractions2atoms(fractions, atoms, fractional_elements, constrained_fractions):
        # element 0 have fraction 1.             element 1 have fraction 0. 
        atoms=atoms.copy()
        
        fmask, n_0 = FractionConverter.set_fractional_atoms(atoms, fractional_elements)        
        transformable_atoms=np.array(fmask)
        transformable_atoms[constrained_fractions]=False
                
        count=0
        if len(constrained_fractions)>0:
            for i in constrained_fractions:
                if atoms.symbols[i]==fractional_elements[0]:
                    count+=1 
        
       # argsort = np.argsort(fractions[fmask])[::-1] 
        argsort = np.argsort(fractions)[::-1]   # argsorting from highest to lowest. 

        for idx in argsort:
            if transformable_atoms[idx]:
                if count <n_0:
                    atoms.symbols[idx]=fractional_elements[0]
                    count+=1
                else:
                    atoms.symbols[idx]=fractional_elements[1]
  
        
        assert list(atoms.symbols).count(fractional_elements[0]) == n_0  
        
        return atoms
 







#################################
class ParallelICE(PostprocessBaseClass, RayPoolUser):

    name = 'ICEPoolRelaxer'
    
#    def __init__(self, model=None, optimizer=None,  optimizer_kwargs={}, kwargs={}):
    def __init__(self, model=None, optimizer=None, start_relax=1, **kwargs):
                 
        self.optimizer=optimizer
        self.model=model        
        
        self.start_relax=start_relax
#        self.optimizer_args=optimizer_kwargs

       # optimizer_run_kwargs={'fmax':0.05, 'steps':200}, fix_template=True,
        
        
       # optimizer_kwargs={'logfile':None}, constraints=[], start_relax=1, 
       # **kwargs):
        
        
        pool_user_kwargs = {key:kwargs.pop(key, None) for key in RayPoolUser.kwargs if kwargs.get(key, None) is not None}
        RayPoolUser.__init__(self, **pool_user_kwargs)
        PostprocessBaseClass.__init__(self, **kwargs)  # erstatter superinit


      #  self.optimizer = ICESurrogateOptimizer()
        
        
        
  #      self.optimizer_kwargs = optimizer_kwargs


 #       self.optimizer_run_kwargs = optimizer_run_kwargs

        print('in init') 
        print(self.model)
        print(self.model.ready_state)
  #      self.start_relax = start_relax
  #      self.model = model
  #      self.constraints = constraints
  #      self.fix_template = fix_template
        self.model_key = self.pool_add_module(model)



    def process_list(self, candidates):
        
        """
        Relaxes the given candidates in parallel using Ray. 

        Parameters
        ----------
        list_of_candidates : listn
            List of AGOX candidates to relax. 

        Returns
        -------
        list
            List of relaxed candidates.
        """
        # Apply constraints.  
      #  [self.apply_constraints(candidate) for candidate in candidates]    # sætter constraints på

        # Make args, kwargs and modules lists:
        N_jobs = len(candidates)
        modules = [[self.model_key]] * N_jobs      # "lægger modellen ud på processorer"
        args = [[candidate] + [self.optimizer] for candidate in candidates]  # argumenter for kørslen
        kwargs = [{} for _ in range(N_jobs)]   # keyword argumenter.  ikke brugt lige nu
        relaxed_candidates = self.pool_map(relax, modules, args, kwargs)   # relax kaldes med modulesm args og kwargs

        [cand.set_positions(relax_cand.positions) for cand, relax_cand in zip(candidates, relaxed_candidates)] 

        
        # Remove constraints & move relaxed positions to input candidates:
        # This is due to immutability of the candidates coming from pool_map.
     #   [self.remove_constraints(candidate) for candidate in candidates]    # fjerner constraints
     #   [cand.set_positions(relax_cand.positions) for cand, relax_cand in zip(candidates, relaxed_candidates)]  # objecter der kommer ud af pool_map kan ikke skrives på (immuteable)
        # for hvert input sættes de relaxerede koordinater tilbage

     #   average_steps = np.mean([candidate.get_meta_information('relaxation_steps') 
     #       for candidate in relaxed_candidates])
     #   self.writer(f'{len(relaxed_candidates)} candidates relaxed for an average of {average_steps} steps.')

        return relaxed_candidates

    def postprocess(self, candidate):
        raise NotImplementedError('"postprocess"-method is not implemented, use postprocess_list.')

    def do_check(self, **kwargs):
        return self.check_iteration_counter(self.start_relax) * self.model.ready_state



def relax(model, candidate, optimizer):
        
    candidate = candidate.copy()
    print('in relax')
    print(model)
    print(model.hp)
    relaxed_candidate=optimizer.relax(candidate, model)
    
    return relaxed_candidate

################################













    
    
class ICESurrogateOptimizer:    
    '''
    Small optimizer class to handle local optimizations in ICEBEACON.
    '''
    
    name = 'ICE'
    
    def __init__(self, # model,
                 fmax=0.05, 
                 relax_steps=100,   # CL I removed gp  as the first argument
                 write_surropt_trajs=False,
                 write_surropt_fracs=False,
                 randomtype='drs',
                 #pos_cindex=[],                # remove this.  its implemented in atoms.constraints
                 frac_cindex=[],
#                 fit_coordinates=True,         # remove this.  set to all instead.  
#                 fit_fractions=True,           # remove this.  set to all instead.
                 fractional_elements=None,
                 Ghost_mode=False,
               #  n_ghost=0,
                 n_real=None,
                 pos_lims=np.array([-1,-1,-1,1,1,1])*np.inf,
                 frac_lims=[0.0,1.0],
                 defractioning_steps=0,
                 defractioning_sub_steps=0,
                 post_rounding_steps=0,
                 derivative_modulation=[1.0,1.0],
                 **kwargs):
      #  super().__init__(**kwargs)   # fordi den ikke arver fra noget
        #if n_ghost>0:
        #    Ghost_mode=True
        
        
        if Ghost_mode and (n_real is None): 
            raise RuntimeError('n_real must be set when running Ghost_mode')
        
        self.n_real=n_real      # O 
        
#        self.model=model        # M
        
        #self.gp = gp
        self.fmax = fmax                     # R
        self.relax_steps = relax_steps       # R
        self.write_surropt_trajs = write_surropt_trajs        # O
        self.write_surropt_fracs = write_surropt_fracs        # O  
#        self.fit_coordinates = fit_coordinates
#        self.fit_fractions=fit_fractions
        #self.pos_cindex = pos_cindex
        self.frac_cindex=frac_cindex      #   O
        self.pos_lims=pos_lims            #   O
        self.frac_lims=frac_lims          #   O
        
        self.defractioning_steps=defractioning_steps              # R
        self.defractioning_sub_steps=defractioning_sub_steps      # R
        self.post_rounding_steps=post_rounding_steps              # R
        self.derivative_modulation=derivative_modulation          # R

        
        self.fractional_elements=fractional_elements  # O
        #self.n_ghost=n_ghost
        self.Ghost_mode=Ghost_mode  # O



        self.rfg = RandomFractionGenerator(frac_cindex=frac_cindex,
                                           Ghost_mode=Ghost_mode,
                                           randomtype=randomtype,
                                           fractional_elements=fractional_elements,
                                           n_real=n_real,
                                           frac_lims=frac_lims)

        #self.index=0

        #self.step_sum=relax_steps+defractioning_steps*defractioning_sub_steps+post_rounding_steps

        if not Ghost_mode and (defractioning_steps>0):                             # GHOST_LOGIC
            raise RuntimeError("defractioning is not meant for ICE-BEACON."
                               'set defractioning_steps and defractioning sub_steps to zero') 


    def get_constrained_atoms(self, atoms):
        pos_cindex = []
        for C in atoms.constraints:
            if isinstance(C, FixAtoms):
                pos_cindex=C.index
                
        return pos_cindex


    def _calculate_properties(self, params, *args):
        '''
        Function to be minimized. Returns the predicted energy and
        its gradients w.r.t. both coordinates and fractions.
        '''        
       
        atoms = args[1]
        natoms = len(atoms)

        coords = params.reshape(natoms, 4)[:, :3]            
        fractions = params.reshape(natoms, 4)[:, 3]         

        atoms = atoms.copy()   
    
        atoms.positions = coords
        
        #######################
        model=args[2]
        
        #fp=model.predict(atoms=atoms, fractions=fractions) 
        
        n_ghost=args[3]
        
        fit_fractions=args[4]        
        
#        results, variance = model.predict(atoms=atoms, 
#                                          fractions=fractions,
#                                          get_variance=False,  
#                                         #    calc_coord_gradients=model.gp.use_forces,  CL: wasnt used. its allready in the information
#                                             calc_frac_gradients=fit_fractions,    # problem. 
#                                             n_ghost=n_ghost)        
#        energy = results[0]
#        derivatives = results[1:]
            
        ##############################
        #Super ugly code in order to 
        
        atoms.fractions=fractions
        atoms.n_ghost=n_ghost
       
        atoms.calc=model 
       
        #model.calculate(atoms)
        
        
        energy=atoms.get_potential_energy()
        derivatives=-1*atoms.get_forces().flatten()
        
        
        #print(energy)
        #print(derivatives)
        #print(np.shape(derivatives))
         
  #      print(energy)
 #       print(derivatives)
        
        # er det vigtigt at der står self.atoms i calculate?
        # hvordan får jeg acquisitoren til at tage imod fractions, så jeg ikke får fejl?
        # hvordan får jeg mine resultater ud så jeg kan bruge dem her. 
        
        ##############################
        
       
        
        
        

        #u = variance        
        #print(energy)
        #print(derivatives)
        #print(np.shape(derivatives))
        
    
        #######################
            
        
        writer = args[0]
        writer.set_properties(energy=energy, 
                              gradients=derivatives.reshape(natoms, 4)[:, :3])
        
        writer.set_atoms(atoms)


        energy, derivatives=self.rescale_output(energy, derivatives)

        ##############
        #energy = energy  *   self.derivative_modulation[0]
        #derivatives = derivatives  * self.derivative_modulation[0]
        #derivatives[3:len(derivatives):4] = derivatives[3:len(derivatives):4]  * (self.derivative_modulation[1]/self.derivative_modulation[0])
        ##############

        return energy , np.array(derivatives)



    def rescale_output(self, energy, derivatives):
        energy = energy  *   self.derivative_modulation[0]
        derivatives = derivatives  * self.derivative_modulation[0]
        derivatives[3:len(derivatives):4] = derivatives[3:len(derivatives):4]  * (self.derivative_modulation[1]/self.derivative_modulation[0])
        return energy, derivatives




    





    def initiate_writer(self, atoms, fractions, model, n_ghost, index, subindex):

        fractions = fractions.reshape(-1, 1)
        params = np.concatenate((atoms.positions, fractions), axis=1).flatten()
        
        
        writer = OptimizationWriter(atoms=atoms,
                                    fractional_elements=self.fractional_elements,
                                    index=index, subindex=subindex,
                                    write_surropt_fracs=self.write_surropt_fracs,
                                    write_surropt_trajs=self.write_surropt_trajs)
        
        
        fit_fractions=True
        
        e, f = self._calculate_properties(params, writer, atoms, model, n_ghost, fit_fractions)
        writer.set_properties(energy=e/self.derivative_modulation[0] , gradients=f/self.derivative_modulation[0])
        writer.set_atoms(atoms)
        #writer.write_atoms(params)
        writer.write_atoms(params)
        return writer



    #def relax(self, atoms, model, fractions, index, subindex):  # CL commented out by me. now this method unpacs the fractions itself.  
    def relax(self, atoms, model):#, model, index=0, subindex=0):  
        
        
#        if not self.model.ready_state:
#            return atoms

        print('Im relaxing')
        print(self.relax_steps)
        print(model.ready_state)
        print(model)
        
        if not model.ready_state:
            return atoms
        
        '''
        Relax atoms in the surrogate potential.
        '''
        
        
        
        '''
        with rfg in relax. 
        if self.Ghost_mode:  # work for both nbest and rattle
            if n0==n_real: 
                atoms2fractions
                constrain all 
                n_ghost=0         
            else:
                rfg.get(atoms)    
        else:  (i.e. if ICE)
            rfg.get()   unless one of best
            
            
            jeg tror aldrig der var noget der forhindrede 
            nbest i at gå væk fra sin initial tilstand?
            
            det ville være en simpel løsning at sige, at ICE altid shuffler
            og at ghost aldrig shuffler. 
            så kan rgen nemt komme ind i surropt som det er nu. 
            og ICEinitatomsgen kan helt slettes
            
        
        '''
        
        
        

        fractions, frac_cindex, n_ghost = self.rfg.setup_fractions(atoms)

        #fractions=atoms.fractions
        
        
        
        #if atoms.from_lob and self.Ghost_mode:
        #    n_ghost=0    
        #    writer=self.initiate_writer(atoms, fractions, model, n_ghost, index, subindex)
        #    success, opt_atoms = self.round_relax(atoms, model, self.step_sum, writer)        
        #    return opt_atoms, success
            
        
        #model=self.model
        
        print('Hello')
        index=0
        subindex=0
        writer=self.initiate_writer(atoms, fractions, model, n_ghost, index, subindex)


        success, opt_atoms, opt_fractions = self.constrain_and_minimize(atoms, model, fractions, self.frac_lims, self.relax_steps, frac_cindex, writer, n_ghost)

        if self.defractioning_steps>0:
            success, opt_atoms, opt_fractions= self.defractionate_relax(opt_atoms, model, opt_fractions, writer, n_ghost, frac_cindex)
            
        
        opt_atoms=self.defractionalize_atoms(opt_atoms, opt_fractions, n_ghost, frac_cindex)

        
        if self.post_rounding_steps>0:
            success, opt_atoms = self.round_relax(opt_atoms, model, self.post_rounding_steps, writer, frac_cindex)
   
        #CL I removed opt_fractions as I just do the defractionalization inside of the surrogate optimizer now
        return opt_atoms#, opt_fractions  



    def defractionate_relax(self, atoms, model, fractions, writer, n_ghost, frac_cindex):
        f_change=self.frac_lims[0]*n_ghost/self.defractioning_steps
        for i in range(self.defractioning_steps):
            lower_lim= self.frac_lims[0] * (self.defractioning_steps-(i+1))/self.defractioning_steps
            
            
            
            nc_atoms=np.delete(np.arange(len(atoms)),frac_cindex)
            fractions[nc_atoms]=fractions[nc_atoms]-np.ones(len(nc_atoms))*(f_change/len(nc_atoms))
            
            #fractions=fractions-np.ones(len(atoms))*(f_change/len(atoms))   # should only be done on nonfractional atoms.
            
            
            
            success, opt_atoms, opt_fractions = self.constrain_and_minimize(atoms, model, fractions, [lower_lim,1], self.defractioning_sub_steps, frac_cindex, writer, n_ghost)
                    
        return success, opt_atoms, opt_fractions 


    def round_relax(self, atoms, model, steps, writer, frac_cindex):
        #print('round relax')
        
        round_fractions=FractionConverter.atoms2fractions(atoms, self.fractional_elements)
        
        frac_cindex=np.arange(len(atoms))
        
        n_ghost=0
         
        success, opt_atoms, opt_fractions = self.constrain_and_minimize(atoms, model, round_fractions, [0,1], steps, frac_cindex, writer, n_ghost)
        
        opt_atoms = FractionConverter.fractions2atoms(opt_fractions,       
                                                      atoms=opt_atoms,
                                                      fractional_elements=self.fractional_elements,
                                                      constrained_fractions=frac_cindex)
        
        return success, opt_atoms






    def defractionalize_atoms(self, opt_atoms, opt_fractions, n_ghost, frac_cindex):

        if self.Ghost_mode:
            fmask, n_0 = FractionConverter.set_fractional_atoms(opt_atoms, self.fractional_elements)
            ghost_mask=GhostHandler.generate_ghosts_constrained(opt_atoms, opt_fractions, n_ghost, frac_cindex, fmask)
            opt_atoms=GhostHandler.construct_processed_atoms_object(opt_atoms, ghost_mask)
        else:
            opt_atoms = FractionConverter.fractions2atoms(opt_fractions,       
                                                          atoms=opt_atoms,
                                                          fractional_elements=self.fractional_elements,
                                                          constrained_fractions=frac_cindex)       
        
        return opt_atoms






    def constrain_and_minimize(self, atoms, model, fractions, frac_lims, steps, frac_cindex, writer, n_ghost):
        
        fractions = fractions.reshape(-1, 1)
        params = np.concatenate((atoms.positions, fractions), axis=1).flatten()
        
        fmask, n_0 = FractionConverter.set_fractional_atoms(atoms, self.fractional_elements)
        
        
        
        pos_cindex=self.get_constrained_atoms(atoms)
        
        #print('in minimize')
        #print(pos_cindex)
        
        linear_constraints = SurrOptConstr.get_constraints(atoms,
                                                           fractions,
                                                           fmask, 
                                                           n_0,
                                                           n_ghost,          # set to 
                                                           #self.pos_cindex,      # atoms.constraints.indices
                                                           pos_cindex,
                                                           frac_cindex,        # This one is troublesome for running best structures. but will be overuled by fit_fractions.
                                                         #  self.fit_coordinates,  # not necessary   better not to have
                                                         #  fit_fractions,         # not necessary,  but nice to have
                                                           self.pos_lims,
                                                           frac_lims)
        with warnings.catch_warnings():
            
            fit_fractions=(len(frac_cindex)<len(atoms))
            
            
            warnings.filterwarnings('ignore', category=OptimizeWarning)
            result = minimize(self._calculate_properties,   
                              params,
                              args=(writer, atoms, model, n_ghost, fit_fractions),          # ad n_ghost here
                              method='SLSQP',
                              constraints=linear_constraints,
                              jac=True,
                              options={'ftol':self.fmax, 'maxiter': steps},
                              callback=writer.write_atoms)
            
            
        success = result['success']
        opt_array = result['x'].reshape(len(atoms), 4)
        atoms.positions=opt_array[:, :3].reshape(-1, 3)
        fractions = opt_array[:, 3].flatten()
        
        return success, atoms, fractions        
   
    


class SurrOptConstr:   # All Atoms.
    '''
    Collection of static methods to build linear constraints for optimization
    with SLSQP minimizer within Scipy.
    '''

    @staticmethod
    def get_constraints(atoms, fractions, fmask, n_0, n_ghost,
                        cindex, frac_cindex, pos_lims, frac_lims):
        
        
        '''
        Get appropriate linear constraints for atoms based on the index of
        constrained atoms ('cindex') and the fraction mask (fmask')
        '''
        
        
        natoms = len(atoms)
           
        
        c = [SurrOptConstr.get_fixed_number_of_atoms(natoms, n_0, fmask, n_ghost, frac_lims)]    
        A, lb, ub = SurrOptConstr.init_arrays(len(atoms))
        lb, ub = SurrOptConstr.constrain_positions(atoms, cindex, pos_lims, lb, ub)
        lb, ub = SurrOptConstr.constrain_fractions(fractions, fmask, frac_cindex, frac_lims, lb, ub)
        c.append(LinearConstraint(A=A,lb=lb,ub=ub))
        return tuple(c)
        
        
    @staticmethod    
    def constrain_positions(atoms, cindex, pos_lims, lb, ub):
        for i in range(len(atoms)):
            if i in cindex:
                lb, ub = SurrOptConstr.set_single_constrained(lb, ub, atoms, i)
            else:
                 lb, ub = SurrOptConstr.set_single_unconstrained(lb, ub, atoms, i, pos_lims)
        return lb, ub
        
        
    @staticmethod    
    def constrain_fractions(fractions, fmask, frac_cindex, frac_lims, lb, ub):
        for i in range(len(fmask)):
            if not fmask[i]:                    # i.e. if fmask False. i.e. atoms not fractional
               # set fractions to zero for non-fractional atoms:
               lb[4 * i + 3] = 0         
               ub[4 * i + 3] = 0
               # constrain fractions
            elif i in frac_cindex:
               lb[4 * i + 3] = fractions[i]
               ub[4 * i + 3] = fractions[i]
            else:
               # unconstrain fractions:
               lb[4 * i + 3] = frac_lims[0]
               ub[4 * i + 3] = frac_lims[1]
        return lb, ub
        
            
    @staticmethod
    def get_fixed_number_of_atoms(natoms, n_0, fmask, n_ghost, frac_lims):
        '''
        Get equality constraints to fix the number of atoms
        of different elements.
        '''
        A = np.zeros(4 * natoms)           
        A[3::4][fmask] = 1
        return LinearConstraint(A=A,
                                lb=n_0-n_ghost + n_ghost*frac_lims[0],
                                ub=n_0-n_ghost + n_ghost*frac_lims[0])


    @staticmethod
    def set_single_constrained(lb, ub, atoms, index):
        '''
        Set single coordinate constraint when atom with index
        'index' is constrained with FixAtoms.
        '''
        i = index
        lb[4 * i: 4 * i + 3] = atoms.positions[i]
        ub[4 * i: 4 * i + 3] = atoms.positions[i]

        return lb, ub  


    @staticmethod
    def set_single_unconstrained(lb, ub, atoms, index, pos_lims):
        '''
        Set single coordinate constraint if atom with index
        'index' is not constrained.

        XXX Debug me. Right now I assume a rectangular unit cell.
        '''
        i = index
        lb[4 * i: 4 * i + 3] = pos_lims[:3]
        ub[4 * i: 4 * i + 3] = pos_lims[3:]
        return lb, ub   
     
        
    @staticmethod
    def init_arrays(natoms):
        A = np.eye(4 * natoms)
        lb, ub = (np.zeros(4 * natoms),
                  np.zeros(4 * natoms))
        return A, lb, ub   


class OptimizationWriter:   
    '''
    Handles output of trajectories and atom fractions.
    
    CL:  note to myself.   right now all cores write to the same file.
    MP says there is no good solution to solve that. 
    only use the write modules when running with one CPU.  
    however the mainfile is also always 000, as index=0.
    in the parallel version, I cant just set selv.index+=1 as the optimizer
    in the parallel map never returns.  it will have to be set as a parameter
    in the ICE_ray main runner. ev.t by importang optimizer_kwargs just like 
    they do in their version og ray_pool_relax. 
    '''
    

    

    def __init__(self, atoms, fractional_elements, index, subindex,
                 write_surropt_fracs=False, write_surropt_trajs=False):
        self.atoms = atoms
        self.natoms = len(atoms)
        self.fractional_elements = fractional_elements
        self.write_surropt_fracs = write_surropt_fracs
        self.write_surropt_trajs = write_surropt_trajs

        if self.write_surropt_fracs:
            self.relaxfile = FileWriter('relax_{:03d}_{:03d}.txt'
                                        .format(index, subindex),
                                        printout=False,
                                        write_time=False)

        if self.write_surropt_trajs:
            self.optfilename = 'opt_{:03d}_{:03d}.xyz'.format(index, subindex)

            # format:
            f = open(self.optfilename, 'w')
            f.close()

    def set_properties(self, energy, gradients):
        self.energy = energy
        self.gradients = gradients


    def set_atoms(self, atoms):
        self.atoms = atoms
        self.natoms = len(atoms)

    def write_atoms(self, params):

        fractions = params.reshape(self.natoms, 4)[:, 3]
       # fractions = fractions.reshape(len(fractions),)
        if self.write_surropt_fracs:
            frac_string = ''.join(['{:6.02f}'.format(f) for f in fractions])
            
            
            self.relaxfile.write('{} {:12.04f}'.format(frac_string,
                                                       self.energy))

        if self.write_surropt_trajs:
            atoms = self.atoms.copy()
            coords = params.reshape(self.natoms, 4)[:, :3]
            atoms.positions = coords




            # converting fractions to atoms.  only for writing.
            # not technically necessary.   only relevant to ICE-BEACON
            atoms = FractionConverter.fractions2atoms(fractions=fractions,
                                                      atoms=atoms,
                                                      fractional_elements=self.fractional_elements,
                                                      constrained_fractions=[])
            
            
            
            results = dict(energy=self.energy,
                           forces=-self.gradients)
            atoms.calc = SinglePointCalculator(atoms, **results)

            with warnings.catch_warnings():

                # with EMT, a warning is triggered while writing the
                # results in ase/io/extxyz.py. Lets filter that out:
                warnings.filterwarnings('ignore', category=UserWarning)


###################################################################
                atoms.set_initial_charges(charges=fractions)
###################################################################
                write(self.optfilename,
                      atoms,
                      append=True,
                      parallel=False)




'''
    def write_atoms(self, params):

        fractions = params.reshape(self.natoms, 4)[:, 3]
        if self.write_surropt_fracs:
            frac_string = ''.join(['{:6.02f}'.format(f) for f in fractions])
            
            
            self.relaxfile.write('{} {:12.04f}'.format(frac_string,
                                                       self.energy))

        if self.write_surropt_trajs:
            atoms = self.atoms.copy()
            coords = params.reshape(self.natoms, 4)[:, :3]
            atoms.positions = coords




            # converting fractions to atoms.  only for writing.
            # not technically necessary.   only relevant to ICE-BEACON
            atoms = FractionConverter.fractions2atoms(fractions=fractions,
                                                      atoms=atoms,
                                                      fractional_elements=self.fractional_elements,
                                                      constrained_fractions=[])
            
            
            
            results = dict(energy=self.energy,
                           forces=-self.gradients)
            atoms.calc = SinglePointCalculator(atoms, **results)

            with warnings.catch_warnings():

                # with EMT, a warning is triggered while writing the
                # results in ase/io/extxyz.py. Lets filter that out:
                warnings.filterwarnings('ignore', category=UserWarning)


###################################################################
                atoms.set_initial_charges(charges=fractions)
###################################################################
                write(self.optfilename,
                      atoms,
                      append=True,
                      parallel=False)
'''




# initatomsgen bruger ikke n_ghost
# i surropt udspringer alt af relax, som har både atoms og n_ghost.n_ghost kan bare laves til en variabel der sættes i starten af relax
# rfg har også atoms og laver n0 i alle de relevante metoder.   n_ghost=n0-n_real



# initialize n_real as number of element 0 in the initial set
# if n_real==n0  -->  n_ghosty=0.    and if ghost_mode.  all fractions should be constrained
#
#
#

