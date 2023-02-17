from agox.models.beacon.beacon_fp  import (BeaconFingerPrint, RadialFP, RadialFPCalculator,
                                                            RadialFPGradientCalculator,
                                                            RadialAngularFP, RadialAngularFPCalculator,
                                                            RadialAngularFPGradientCalculator,
                                                            AtomPairs, AtomTriples,
                                                            FPElements)

import numpy as np


class FractionalBeaconFingerPrint(BeaconFingerPrint):
    
    
    def __init__(self, fp_args={}, fractional_elements=None, Ghost_mode=False,
                 weight_by_elements=False,
                 calc_coord_gradients=True,
                 calc_frac_gradients=True,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.fp_args=fp_args
        self.Ghost_mode=Ghost_mode
        self.fractional_elements=fractional_elements
        self.weight_by_elements=weight_by_elements
        self.calc_coord_gradients=calc_coord_gradients
        self.calc_frac_gradients=calc_frac_gradients        
    
    
    
        
    
    def get_fp_object(self, atoms, fractions=None):   
        
            fmask = [(symbol in self.fractional_elements)
                     for symbol in atoms.get_chemical_symbols()]
                
                
            if fractions is None:
                fractions = FPFractions.get_full_fractions(atoms, self.fractional_elements)
                
        
            fp_object=PartialRadAngFP(atoms=atoms, 
                                      fractions=fractions,
                                      elements=self.fractional_elements,
                                      fmask=fmask,
                                      calc_coord_gradients=self.calc_coord_gradients,
                                      calc_frac_gradients=self.calc_frac_gradients,
                                      weight_by_elements=self.weight_by_elements,
                                      fractional_elements=self.fractional_elements,
                                      Ghost_mode=self.Ghost_mode,
                                      **self.fp_args)
           
            return fp_object
        
        
        

    def create_global_features(self, atoms):
        
        fp_object=self.get_fp_object(atoms)
        
        feature=fp_object.vector
        
        return  feature



    def create_global_feature_gradient(self, atoms):
        
        fp_object=self.get_fp_object(atoms)
        
        feature_gradients=fp_object.reduce_coord_gradients()
        
        return feature_gradients        
        


class PartialFP(RadialFP):

    def __init__(self, atoms=None, fractions=None,
                 calc_coord_gradients=True,
                 calc_frac_gradients=True,
                 fmask=None, Ghost_mode=False,         # CL I added n_ghost
                 elements=None, **kwargs):


        #print('I am here')
        '''
        fractions: array(Natoms) fraction for each atom
        fmask:     array(Natoms) boolean for each atom;
                                 if fmask[i] == True, then atom i is fractional
        elements: array(2) elements which are fractionized
        '''

        default_parameters = {'r_cutoff': 8.0,
                              'r_delta': 0.4,
                              'r_nbins': 200}
    
        self.params = default_parameters.copy()
        self.params.update(kwargs)

    

        #self.calc_gradients = calc_gradients
        
        ######
        #calc_coord_gradients=calc_gradients
        ######
        
        #self.calc_frac_gradients = calc_frac_gradients

        # parameters in this class for constructing the fingerprint:
        self.param_names = ['r_cutoff', 'r_delta', 'r_nbins']


        if elements is None:
            elements = np.sort(list(set(atoms.symbols)))

           
            if (len(elements) !=2  and  (not Ghost_mode)):
                raise RuntimeError('System has more than 2 elements. '
                                   'Fractionized elements need to '
                                   'be specified.')

        elif (len(elements) != 2 and (not Ghost_mode)):
            raise RuntimeError('Too many elements specified. '
                               'Only 2 elements supported.')
        
        elif (len(elements)>1 and Ghost_mode):
            raise RuntimeError('ghost run with more than one element not supported')
                               

        if fmask is None:
            fmask = [(symbol in elements)
                     for symbol in atoms.get_chemical_symbols()]

        if len(atoms) != len(fmask):
            
            raise RuntimeError('Number of fractions must match '
                               'the number of fractioned indices.')

        self.atoms = atoms.copy()
        self.atoms.wrap()

        if fractions is None:
            fractions = FPFractions.get_full_fractions(atoms, elements)


        #self.fractions=fractions  # CL made by me. to make it possible to import into prior


        pairs = AtomPairs(self.atoms,
                          self.params['r_cutoff'])
        
        
        self.pairs = pairs

        frac = FPFractions(self.atoms, fractions, fmask, elements)
        
        
        
#################################################################################### 
                
        if Ghost_mode:
            factors = FPFractions.get_ghost_products_for_pairs(pairs,frac, calc_frac_gradients)              # multiplies fractions. only need to work on pairs where both fractions are nonzero
        else:
            factors = FPFractions.get_fraction_products_for_pairs(pairs,frac, calc_frac_gradients)
                            
#####################################################################################


        fpparams = dict(pairs=pairs,
                        factors=factors,
                        cutoff=self.params['r_cutoff'],
                        width=self.params['r_delta'],
                        nbins=self.params['r_nbins'])

        self.rho_R = PartialFPCalculator.calculate(**fpparams)
        
        #self.rho_R=FP_normalization.weight_array_by_fraction(self.rho_R,fractions)
        
        self.vector = self.rho_R.flatten()

        self.gradients = (PartialFPGradientCalculator.
                          calculate(natoms=self.natoms, **fpparams, calc_coord_gradients=calc_coord_gradients))
                          #if self.calc_gradients else None)
                          
                          
                          
                          
        #print('shake and bake')                           
        #print(np.shape(self.gradients))
        #self.gradients = np.moveaxis(self.gradients, 0, 2)  # reorganize
        #print(np.shape(self.gradients))
        #self.gradients = FP_normalization.weight_array_by_fraction(self.gradients,fractions)
        #print(np.shape(self.gradients))
        #self.gradients = np.moveaxis(self.gradients, 2, 0)  # back-reorganize
        #print(np.shape(self.gradients))  
        #print('scones')                          
                          
                          
##################################################################################
    
    
        if Ghost_mode:                
            gradient_calculator=PartialFPFractionGradientCalculator.calculate_ghost
        else: 
            gradient_calculator=PartialFPFractionGradientCalculator.calculate
            
            
        self.frac_gradients = (gradient_calculator(self.natoms, pairs,
                                             frac,
                                             cutoff=self.params['r_cutoff'],
                                             width=self.params['r_delta'],
                                             nbins=self.params['r_nbins'],
                                             calc_frac_gradients=calc_frac_gradients))
                                   #if self.calc_gradients else None)
        #print('shake and bake')                           
        #print(np.shape(self.frac_gradients))
        #self.frac_gradients = np.moveaxis(self.frac_gradients, 0, 1)  # reorganize
        #print(np.shape(self.frac_gradients))
        #self.frac_gradients = FP_normalization.weight_array_by_fraction(self.frac_gradients,fractions)
        #print(np.shape(self.frac_gradients))
        #self.frac_gradients = np.moveaxis(self.frac_gradients, 1, 0)  # back-reorganize
        #print(np.shape(self.frac_gradients))  
        #print('scones')
##################################################################################
    @property
    def natoms(self):
        return len(self.atoms)

    def reduce_frac_gradients(self):
        return self.frac_gradients.reshape(self.natoms, -1, 1)


class PartialFPCalculator(RadialFPCalculator):

    @classmethod
    def calculate(cls, pairs, factors, cutoff, width, nbins):
        '''
        Calculate the Gaussian-broadened fingerprint.
        '''

        if pairs.empty:
            return np.zeros([pairs.elem.nelem, pairs.elem.nelem, nbins])

        # Gaussians with correct heights:
        gs = cls.get_gs(pairs=pairs,
                         width=width,
                         cutoff=cutoff,
                         nbins=nbins)
        gs *= cls.get_peak_heights(pairs=pairs,
                                    cutoff=cutoff,
                                    nbins=nbins)[:, np.newaxis]

        rho_R = np.einsum('hij,hl->ijl', factors, gs, optimize=True)

        return rho_R

class PartialFPGradientCalculator(PartialFPCalculator,                       # only need pairs where both fractions are nonzero
                                  RadialFPGradientCalculator):

    @classmethod
    def calculate(cls, natoms, pairs, factors, cutoff, width, nbins, calc_coord_gradients):                     
        '''
        Calculate fingerprint gradients with respect to atomic positions.
        '''
        gradients = np.zeros([natoms, pairs.elem.nelem,
                              pairs.elem.nelem, nbins, 3])
                
        if not calc_coord_gradients:
            return gradients
                
        if pairs.empty:
            return gradients

        results = np.einsum('ij,ik->ijk',
                            cls.get_gradient_gaussians(pairs, cutoff,
                                                        nbins, width),
                            -pairs.rm,
                            optimize=True)

        for p in range(len(pairs.indices)):
            i, j_ = pairs.indices[p]
            j = j_ % natoms

            gradients[i] += np.einsum('ij,kl->ijkl',
                                       factors[p],
                                       results[p],
                                       optimize=False)
            gradients[j] += -np.einsum('ij,kl->ijkl',
                                         factors[p],
                                         results[p],
                                         optimize=False)

        return gradients


class PartialFPFractionGradientCalculator(PartialFPCalculator):

    @classmethod
    def calculate(cls, natoms, pairs, fractions,
                  cutoff, width, nbins, calc_frac_gradients):

        f_gradients = np.zeros([natoms, pairs.elem.nelem, pairs.elem.nelem,
                              nbins, 1])
        
        if not calc_frac_gradients:  
            return f_gradients         

        if pairs.empty:
            return f_gradients

        gs = (cls.get_peak_heights(pairs, cutoff, nbins)[:, np.newaxis] *
              cls.get_gs(pairs, width, cutoff, nbins))

        fmask = fractions.fmask
        q = fractions.fractions
        C, D = fractions.ei

        for p, (i, j_) in enumerate(pairs.indices):
            j = j_ % natoms

            if not fmask[i] or not fmask[j]:
                A, B = pairs.elem.indices[p]

                if fmask[i]:  # if atom i is fractioned

                    f_gradients[i, C, B, :, 0] += gs[p]
                    f_gradients[i, D, B, :, 0] += - gs[p]
                    f_gradients[i, B, C, :, 0] += gs[p]
                    f_gradients[i, B, D, :, 0] += - gs[p]

            else:  # if both i and j are fractioned
                f_gradients[i, C, C, :, 0] += 2 * q[j] * gs[p]
                f_gradients[i, C, D, :, 0] += (1 - 2 * q[j]) * gs[p]
                f_gradients[i, D, C, :, 0] += (1 - 2 * q[j]) * gs[p]
                f_gradients[i, D, D, :, 0] += - 2 * (1 - q[j]) * gs[p]

        return f_gradients
    
    
    
    
    @classmethod
    def calculate_ghost(cls, natoms, pairs, fractions,                  # for pair i an j.  only gives something for i, if j is nonzero and vice versa
                  cutoff, width, nbins, calc_frac_gradients):
               
        #C, D = fractions.ei
        
        
        #print(C)
        #print(D)
        
        f_gradients = np.zeros([natoms, pairs.elem.nelem, pairs.elem.nelem,
                              nbins, 1])
                
        if not calc_frac_gradients:  
            return f_gradients 
                            
        if pairs.empty:
            return f_gradients

        gs = (cls.get_peak_heights(pairs, cutoff, nbins)[:, np.newaxis] *
              cls.get_gs(pairs, width, cutoff, nbins))
        
        fmask = fractions.fmask
        q = fractions.fractions
        C = fractions.ei
        
        
        for p, (i, j_) in enumerate(pairs.indices):              # I dont know. det er mega forvirrende. 
            j = j_ % natoms
            A, B = pairs.elem.indices[p]

            if not fmask[i] or not fmask[j]:
                A, B = pairs.elem.indices[p]

                if fmask[i]:  # if atom i is fractioned         why not if fmask[j]
                    f_gradients[i, C, B, :, 0] += gs[p]
                    
                elif fmask[j]:    
                    f_gradients[j, A, C, :, 0] += gs[p]
                    

            else:  # if both i and j are fractioned
                f_gradients[i, C, C,:,0] += q[j] * gs[p]  # same procedure as angular part. 
                f_gradients[j, C, C,:,0] += q[i] * gs[p]
        return f_gradients        
        
        
        
        
        for p, (i, j_) in enumerate(pairs.indices):
            j = j_ % natoms
            A, B = pairs.elem.indices[p]
            
                                    
            f_gradients[i, A, B,:,0] += q[j] * gs[p]  # same procedure as angular part. 
            f_gradients[j, A, B,:,0] += q[i] * gs[p]
        
        return f_gradients






class PartialRadAngFP(PartialFP, RadialAngularFP):

    def __init__(self, atoms=None, fractions=None,
                 calc_coord_gradients=True,
                 calc_frac_gradients=True,
                 fmask=None, Ghost_mode=False,
                 elements=None, **kwargs):
        
        '''
        fractions: array(Natoms) fraction for each atom
        fmask:     array(Natoms) boolean for each atom;
                                 if fmask[i] == True, then atom i is fractional
        elements: array(2) elements which are fractionized
        '''
        #print('but now Im here')
####################################################################################################################
        PartialFP.__init__(self, atoms, fractions,
                           fmask=fmask,
                           Ghost_mode=Ghost_mode,               # CL I added Ghost_mode
                           elements=elements,
                           calc_coord_gradients=calc_coord_gradients,
                           calc_frac_gradients=calc_frac_gradients,
                           **kwargs)
####################################################################################################################

        default_parameters = {'r_cutoff': 8.0,
                             'r_delta': 0.4,
                              'r_nbins': 200,
                              'a_cutoff': 4.0,
                              'a_delta': 0.4,
                              'a_nbins': 100,
                              'gamma': 0.5,
                              'aweight': 1.0}

        

        self.params = default_parameters.copy()
        self.params.update(kwargs)
        

        # parameters in this class for constructing the fingerprint:
        self.param_names = ['r_cutoff', 'r_delta', 'r_nbins',
                            'a_cutoff', 'a_delta', 'a_nbins',
                            'aweight']

        assert self.params['r_cutoff'] >= self.params['a_cutoff']

        triples = AtomTriples(self.atoms,
                              cutoff=self.params['r_cutoff'],
                              cutoff2=self.params['a_cutoff'])
        
        
        self.triples = triples

        if elements is None:
            elements = np.sort(list(set(atoms.symbols)))

        if fractions is None:
            fractions = FPFractions.get_full_fractions(atoms, elements)

        if fmask is None:
            fmask = [(symbol in elements)
                     for symbol in atoms.get_chemical_symbols()]


        frac = FPFractions(self.atoms, fractions, fmask, elements)
        
##############################################################################################   
        
        if Ghost_mode:
            factors = FPFractions.get_ghost_products_for_triples(triples,frac, calc_frac_gradients)
        else:
            factors = FPFractions.get_fraction_products_for_triples(triples,frac, calc_frac_gradients)
                                                                    
##############################################################################################
        
        
        fpparams = dict(triples=triples,
                        factors=factors,
                        cutoff=self.params['a_cutoff'],
                        width=self.params['a_delta'],
                        nbins=self.params['a_nbins'],
                        aweight=self.params['aweight'],
                        gamma=self.params['gamma'])

        self.rho_a = PartialRadAngFPCalculator.calculate(**fpparams)
        
        #       if weight_by_fraction:
        #    print('WORK')
        #self.rho_a=FP_normalization.weight_array_by_fraction(self.rho_a,fractions)
        
        self.vector = np.concatenate((self.rho_R.flatten(),
                                      self.rho_a.flatten()), axis=None)
        
        


        #calc_gradients = (self.calc_gradients if calc_gradients is None
        #                  else calc_gradients)

        
        #calc_coord_gradients=self.calc_coord_gradients
 

        self.anglegradients = (PartialRadAngFPGradientCalculator.
                               calculate(natoms=self.natoms,
                                         **fpparams,   calc_coord_gradients=calc_coord_gradients))
                               #if calc_gradients else None)

        #print('shake and bake')                           
        #print(np.shape(self.anglegradients))
        #self.anglegradients = np.moveaxis(self.anglegradients, 0, 3)  # reorganize
        #print(np.shape(self.anglegradients))
        #self.anglegradients = FP_normalization.weight_array_by_fraction(self.anglegradients,fractions)
        #print(np.shape(self.anglegradients))
        #self.anglegradients = np.moveaxis(self.anglegradients, 3, 0)  # back-reorganize
        #print(np.shape(self.anglegradients))  
        #print('scones')                      

#####################################################################################################
                      
                        
        if Ghost_mode:
            gradient_calculator=PartialRadAngFPFractionGradientCalculator.calculate_ghost
        else:
            gradient_calculator= PartialRadAngFPFractionGradientCalculator.calculate
            
           
        self.frac_gradients_angles = (gradient_calculator(self.natoms,
                                                    triples,
                                                    frac,
                                                    cutoff=self.params['a_cutoff'],
                                                    width=self.params['a_delta'],
                                                    nbins=self.params['a_nbins'],
                                                    aweight=self.params['aweight'],
                                                    gamma=self.params['gamma'],
                                                    calc_frac_gradients=calc_frac_gradients))
                                          #if self.calc_gradients else None)            


        #print('shake and bake')                           
        #print(np.shape(self.frac_gradients_angles))
        #self.frac_gradients_angles = np.moveaxis(self.frac_gradients_angles, 0, 1)  # reorganize
        #print(np.shape(self.frac_gradients_angles))
        #self.frac_gradients_angles = FP_normalization.weight_array_by_fraction(self.frac_gradients_angles,fractions)
        #print(np.shape(self.frac_gradients_angles))
        #self.frac_gradients_angles = np.moveaxis(self.frac_gradients_angles, 1, 0)  # back-reorganize
        #print(np.shape(self.frac_gradients_angles))  
        #print('scones')  
       
        
######################################################################################################                                      

    def reduce_frac_gradients(self):

        return np.concatenate((self.frac_gradients.reshape(self.natoms, -1, 1),
                               self.frac_gradients_angles.reshape(self.natoms, -1, 1)),
                              axis=1)



class PartialRadAngFPCalculator(RadialAngularFPCalculator):

    @classmethod
    def calculate(cls, triples, factors, cutoff, width,
                  nbins, aweight, gamma):

        rho_a = np.zeros([triples.elem.nelem, triples.elem.nelem,
                          triples.elem.nelem, nbins])

        if triples.empty:
            return rho_a

        # Gaussians with correct height:
        gs = cls.get_cutoff_ags(triples, cutoff, width, nbins, aweight, gamma)
        rho_a = np.einsum('hijk,hl->ijkl', factors, gs)

        return rho_a


class PartialRadAngFPGradientCalculator(RadialAngularFPGradientCalculator):

    @classmethod
    def calculate(cls, natoms, triples, factors, cutoff,
                  width, nbins, aweight, gamma, calc_coord_gradients):

        gradients = np.zeros([natoms, triples.elem.nelem, triples.elem.nelem,
                              triples.elem.nelem, nbins, 3])

        if not calc_coord_gradients:
            return gradients


        if triples.empty:
            return gradients

        firsts, seconds, thirds = cls.do_anglegradient_math(triples, cutoff,
                                                            width, nbins,
                                                            aweight, gamma)

        for p in range(len(triples.indices)):
            i, j_, k_ = triples.indices[p]
            j = j_ % natoms
            k = k_ % natoms

            # derivative w.r.t. atom i:
            gradients[i] += np.einsum('ijk,lm->ijklm',
                                      factors[p],
                                      firsts[p],
                                      optimize=False)

            # derivative w.r.t. atom j:
            gradients[j] += np.einsum('ijk,lm->ijklm',
                                      factors[p],
                                      seconds[p],
                                      optimize=False)

            # derivative w.r.t. atom k:
            gradients[k] += np.einsum('ijk,lm->ijklm',
                                      factors[p],
                                      thirds[p],
                                      optimize=False)

        return gradients



class PartialRadAngFPFractionGradientCalculator(RadialAngularFPCalculator):

    @classmethod
    def calculate(cls, natoms, triples, fractions,
                  cutoff, width, nbins, aweight, gamma, calc_frac_gradients):


        f_gradients = np.zeros([natoms,
                                triples.elem.nelem,
                                triples.elem.nelem,
                                triples.elem.nelem,
                                nbins])

        if not calc_frac_gradients:  
            return f_gradients 

        if triples.empty:
            return f_gradients

        gs = cls.get_cutoff_ags(triples, cutoff, width, nbins, aweight, gamma)
        
        #CL This is not the same factors as above for calculating coord-gradients and fingerprint. 
        # this is specific to the angular fraction gradient solvers
        factors = cls.get_factor_product_gradients_for_triples(natoms,  
                                                               triples,
                                                               fractions)

        D, E = fractions.ei
        result = np.einsum('pijk,pl->pijkl',
                           factors,
                           gs,
                           optimize=True)


        for p, (i, j_, k_) in enumerate(triples.indices):
            j = j_ % natoms
            k = k_ % natoms
            r0, r1, r2 = result[p]

            f_gradients[i, D, :, :] += r0
            f_gradients[i, E, :, :] += -r0

            f_gradients[j, :, D, :] += r1
            f_gradients[j, :, E, :] += -r1

            f_gradients[k, :, :, D] += r2
            f_gradients[k, :, :, E] += -r2

        return f_gradients



    @staticmethod
    def get_factor_product_gradients_for_triples(natoms, triples, fractions):

        factors = np.zeros((len(triples.indices), 3,
                            triples.elem.nelem, triples.elem.nelem))

        fmask = fractions.fmask
        q = fractions.fractions
        D, E = fractions.ei

        for p in range(len(triples.indices)):

            i, j_, k_ = triples.indices[p]
            j = j_ % natoms
            k = k_ % natoms
            A, B, C = triples.elem.indices[p]

            if (fmask[i] and
                not fmask[j] and
                not fmask[k]):
                factors[p, 0, B, C] += 1.0

            elif (not fmask[i] and
                  fmask[j] and
                  not fmask[k]):
                factors[p, 1, A, C] += 1.0

            elif (not fmask[i] and
                  not fmask[j] and
                  fmask[k]):
                factors[p, 2, A, B] += 1.0

            elif (not fmask[i] and
                  fmask[j] and
                  fmask[k]):
                factors[p, 1, A, D] += q[k]
                factors[p, 1, A, E] += 1 - q[k]

                factors[p, 2, A, D] += q[j]
                factors[p, 2, A, E] += 1 - q[j]

            elif (fmask[i] and
                  fmask[j] and
                  not fmask[k]):
                factors[p, 0, D, C] += q[j]
                factors[p, 0, E, C] += 1 - q[j]

                factors[p, 1, D, C] += q[i]
                factors[p, 1, E, C] += 1 - q[i]

            elif (fmask[i] and
                  not fmask[j] and
                  fmask[k]):

                factors[p, 0, B, D] += q[k]
                factors[p, 0, B, E] += 1 - q[k]

                factors[p, 2, D, B] += q[i]
                factors[p, 2, E, B] += 1 - q[i]

            elif (fmask[i] and
                  fmask[j] and
                  fmask[k]):

                factors[p, 0, D, D] += q[j] * q[k]
                factors[p, 0, D, E] += q[j] * (1 - q[k])
                factors[p, 0, E, D] += (1 - q[j]) * q[k]
                factors[p, 0, E, E] += (1 - q[j]) * (1 - q[k])

                factors[p, 1, D, D] += q[i] * q[k]
                factors[p, 1, D, E] += q[i] * (1 - q[k])
                factors[p, 1, E, D] += (1 - q[i]) * q[k]
                factors[p, 1, E, E] += (1 - q[i]) * (1 - q[k])

                factors[p, 2, D, D] += q[i] * q[j]
                factors[p, 2, D, E] += q[i] * (1 - q[j])
                factors[p, 2, E, D] += (1 - q[i]) * q[j]
                factors[p, 2, E, E] += (1 - q[i]) * (1 - q[j])

        return factors

    
    
#    @classmethod
#    def calculate_ghost(cls, natoms, triples, fractions,
#                  cutoff, width, nbins, aweight, gamma, calc_frac_gradients):
        
#        f_gradients = np.zeros([natoms,
#                                triples.elem.nelem,
#                                triples.elem.nelem,
#                                triples.elem.nelem,
#                               nbins])


#        if not calc_frac_gradients:
#            return f_gradients


#        if triples.empty:
#            return f_gradients

#        gs = cls.get_cutoff_ags(triples, cutoff, width, nbins, aweight, gamma)


#        q=fractions.fractions

#        for p, (i, j_, k_) in enumerate(triples.indices):
#            j=j_ % natoms
#            k=k_ % natoms
#            A, B, C = triples.elem.indices[p]
            
#            f_gradients[i, A, B, C, :] += q[j] * q[k] * gs[p]
#            f_gradients[j, A, B, C, :] += q[i] * q[k] * gs[p]
#            f_gradients[k, A, B, C, :] += q[i] * q[j] * gs[p]
            
            
#        return f_gradients



    @classmethod
    def calculate_ghost(cls, natoms, triples, fractions,
                  cutoff, width, nbins, aweight, gamma, calc_frac_gradients):
        
        f_gradients = np.zeros([natoms,
                                triples.elem.nelem,
                                triples.elem.nelem,
                                triples.elem.nelem,
                                nbins])


        if not calc_frac_gradients:
            return f_gradients


        if triples.empty:
            return f_gradients

        gs = cls.get_cutoff_ags(triples, cutoff, width, nbins, aweight, gamma)


        fmask = fractions.fmask
        q = fractions.fractions
        D = fractions.ei

        for p, (i, j_, k_) in enumerate(triples.indices):
            j=j_ % natoms
            k=k_ % natoms
            A, B, C = triples.elem.indices[p]
            
#            f_gradients[i, A, B, C, :] += q[j] * q[k] * gs[p]
#            f_gradients[j, A, B, C, :] += q[i] * q[k] * gs[p]
#            f_gradients[k, A, B, C, :] += q[i] * q[j] * gs[p]

            if (fmask[i] and
                not fmask[j] and
                not fmask[k]):
                f_gradients[i, D, B, C, :] += gs[p]
                #f_gradients[j, A, B, C, :] += q[i] * q[k] * gs[p]
                #f_gradients[k, A, B, C, :] += q[i] * q[j] * gs[p]                 
                
            elif (not fmask[i] and
                  fmask[j] and
                  not fmask[k]):
               # f_gradients[i, A, B, C, :] += q[j] * q[k] * gs[p]
                f_gradients[j, A, D, C, :] += gs[p]    #qi --> 1 ?
               # f_gradients[k, A, B, C, :] += q[i] * q[j] * gs[p] 

            elif (not fmask[i] and
                  not fmask[j] and
                  fmask[k]):
               # f_gradients[i, A, B, C, :] += q[j] * q[k] * gs[p]
               # f_gradients[j, A, B, C, :] += q[i] * q[k] * gs[p]
                f_gradients[k, A, B, D, :] += gs[p] 

            elif (not fmask[i] and
                  fmask[j] and
                  fmask[k]):
              #  f_gradients[i, A, B, C, :] += q[j] * q[k] * gs[p]
                f_gradients[j, A, D, D, :] += q[k] * gs[p]
                f_gradients[k, A, D, D, :] += q[j] * gs[p] 

            elif (fmask[i] and
                  fmask[j] and
                  not fmask[k]):
                f_gradients[i, D, D, C, :] += q[j] * gs[p]
                f_gradients[j, D, D, C, :] += q[i] * gs[p]
                #f_gradients[k, A, B, C, :] += q[i] * q[j] * gs[p] 

            elif (fmask[i] and
                  not fmask[j] and
                  fmask[k]):
                f_gradients[i, D, B, D, :] += q[k] * gs[p]
                #f_gradients[j, A, B, C, :] += q[i] * q[k] * gs[p]
                f_gradients[k, D, B, D, :] += q[i] * gs[p] 

            elif (fmask[i] and
                  fmask[j] and
                  fmask[k]):
                f_gradients[i, D, D, D, :] += q[j] * q[k] * gs[p]
                f_gradients[j, D, D, D, :] += q[i] * q[k] * gs[p]
                f_gradients[k, D, D, D, :] += q[i] * q[j] * gs[p]                
                
        return f_gradients










class FPFractions:

    def __init__(self, atoms, fractions, fmask, felements):
        self.atoms = atoms
        self.fractions = self.set_fractions_for_not_fmask(fractions, fmask)
        self.fmask = fmask
        self.ei = [list(FPElements.get_elementset(atoms)).index(e)
                   for e in felements]

    @staticmethod
    def set_fractions_for_not_fmask(fractions, fmask):
        return [(fractions[i] if fmask[i] else 1.0)
                for i in range(len(fractions))]

    @staticmethod
    def get_fraction_products_for_pairs(pairs, fractions, calc_frac_gradients):

        factors = np.zeros((len(pairs.indices), pairs.elem.nelem,
                            pairs.elem.nelem))

        fmask = fractions.fmask  # alias
        q = fractions.fractions  # alias
        natoms = len(q)
        C, D = fractions.ei  # element indices for fractionized elements

        for p in range(len(pairs.indices)):
            i, j_ = pairs.indices[p]
            j = j_ % natoms

            if not fmask[i] or not fmask[j]:
                A, B = pairs.elem.indices[p]

                if fmask[i]:
                    factors[p, C, B] += q[i]
                    factors[p, D, B] += 1 - q[i]

                if fmask[j]:
                    factors[p, A, C] += q[j]
                    factors[p, A, D] += 1 - q[j]

                if (not fmask[i]) and (not fmask[j]):
                    factors[p, A, B] += 1

            else:
                factors[p, C, C] += q[i] * q[j]
                factors[p, C, D] += q[i] * (1 - q[j])
                factors[p, D, C] += (1 - q[i]) * q[j]
                factors[p, D, D] += ((1 - q[i]) *
                                     (1 - q[j]))
                
                  
        return factors

    @staticmethod
    def get_ghost_products_for_pairs(pairs, fractions, calc_frac_gradients): 

        #if not calc_frac_gradients:
        #factors = np.ones((len(pairs.indices), pairs.elem.nelem, pairs.elem.nelem))
         #   return factors
                              
        factors = np.zeros((len(pairs.indices), pairs.elem.nelem, pairs.elem.nelem))


        fmask = fractions.fmask  # alias        
        q = fractions.fractions 
        natoms = len(q)
        C = fractions.ei  # element indices for fractionized elements        

        for p in range(len(pairs.indices)):
            i, j_ = pairs.indices[p]
            j = j_ % natoms

            if not fmask[i] or not fmask[j]:
                A, B = pairs.elem.indices[p]
                
                if fmask[i]:
                    factors[p, C, B] += q[i]

                if fmask[j]:
                    factors[p, A, C] += q[j]

                if (not fmask[i]) and (not fmask[j]):
                    factors[p, A, B] += 1

            else:
                factors[p, C, C] += q[i] * q[j]
        
        return factors 



    @staticmethod
    def get_fraction_products_for_triples(triples, fractions, calc_frac_gradients):
        '''
        Precompute all products between fractions and
        extended fractions for atoms in triples.
        '''
        #if not calc_frac_gradients:
        factors = np.zeros((len(triples.indices), triples.elem.nelem,
                            triples.elem.nelem, triples.elem.nelem))

        fmask = fractions.fmask  # alias
        q = fractions.fractions  # alias
        natoms = len(q)
        D, E = fractions.ei  # element indices for fractionized elements.   # this is getting overwritten

        for p in range(len(triples.indices)):
            i, j_, k_ = triples.indices[p]
            j = j_ % natoms
            k = k_ % natoms
            A, B, C = triples.elem.indices[p]
            #D, E = fractions.ei

            if (not fmask[i] and
                not fmask[j] and
                not fmask[k]):
                factors[p, A, B, C] += 1

            elif (fmask[i] and
                  not fmask[j] and
                  not fmask[k]):
                factors[p, D, B, C] += q[i]
                factors[p, E, B, C] += 1 - q[i]

            elif (not fmask[i] and
                  fmask[j] and
                  not fmask[k]):
                factors[p, A, D, C] += q[j]
                factors[p, A, E, C] += 1 - q[j]

            elif (not fmask[i] and
                  not fmask[j] and
                  fmask[k]):
                factors[p, A, B, D] += q[k]
                factors[p, A, B, E] += 1 - q[k]

            elif (fmask[i] and
                  fmask[j] and
                  not fmask[k]):

                factors[p, D, D, C] += q[i] * q[j]
                factors[p, D, E, C] += q[i] * (1 - q[j])
                factors[p, E, D, C] += (1 - q[i]) * q[j]
                factors[p, E, E, C] += (1 - q[i]) * (1 - q[j])

            elif (fmask[i] and
                  not fmask[j] and
                  fmask[k]):

                factors[p, D, B, D] += q[i] * q[k]
                factors[p, D, B, E] += q[i] * (1 - q[k])
                factors[p, E, B, D] += (1 - q[i]) * q[k]
                factors[p, E, B, E] += (1 - q[i]) * (1 - q[k])

            elif (not fmask[i] and
                  fmask[j] and
                  fmask[k]):
                factors[p, A, D, D] += q[j] * q[k]
                factors[p, A, D, E] += q[j] * (1 - q[k])
                factors[p, A, E, D] += (1 - q[j]) * q[k]
                factors[p, A, E, E] += (1 - q[j]) * (1 - q[k])

            elif (fmask[i] and
                  fmask[j] and
                  fmask[k]):
                factors[p, D, D, D] += q[i] * q[j] * q[k]
                factors[p, D, D, E] += q[i] * q[j] * (1 - q[k])
                factors[p, D, E, D] += q[i] * (1 - q[j]) * q[k]
                factors[p, D, E, E] += q[i] * (1 - q[j]) * (1 - q[k])
                factors[p, E, D, D] += (1 - q[i]) * q[j] * q[k]
                factors[p, E, D, E] += (1 - q[i]) * q[j] * (1 - q[k])
                factors[p, E, E, D] += (1 - q[i]) * (1 - q[j]) * q[k]
                factors[p, E, E, E] += (1 - q[i]) * (1 - q[j]) * (1 - q[k])

        return factors
    
#    @staticmethod
#    def get_ghost_products_for_triples(triples, fractions, calc_frac_gradients):
  
#        if not calc_frac_gradients:
#            factors = np.ones((len(triples.indices), triples.elem.nelem, triples.elem.nelem, triples.elem.nelem))
#            return factors
        
        
#        factors = np.zeros((len(triples.indices), triples.elem.nelem,
#                            triples.elem.nelem, triples.elem.nelem))

#        q = fractions.fractions
#        natoms = len(q)

#        for p in range(len(triples.indices)):
#            i, j_, k_ = triples.indices[p]
#            j = j_ % natoms
#            k = k_ % natoms
#            A, B, C = triples.elem.indices[p]

#            factors[p, A, B, C] += q[i] * q[j] * q[k]   

#        return factors


    @staticmethod
    def get_ghost_products_for_triples(triples, fractions, calc_frac_gradients):
        '''
        Precompute all products between fractions and
        extended fractions for atoms in triples.
        '''
        #if not calc_frac_gradients:
        factors = np.zeros((len(triples.indices), triples.elem.nelem,
                            triples.elem.nelem, triples.elem.nelem))

        fmask = fractions.fmask  # alias
        q = fractions.fractions  # alias
        natoms = len(q)
        D = fractions.ei  # element indices for fractionized elements.   # this is getting overwritten

        for p in range(len(triples.indices)):
            i, j_, k_ = triples.indices[p]
            j = j_ % natoms
            k = k_ % natoms
            A, B, C = triples.elem.indices[p]
            #D, E = fractions.ei

            if (not fmask[i] and
                not fmask[j] and
                not fmask[k]):
                factors[p, A, B, C] += 1

            elif (fmask[i] and
                  not fmask[j] and
                  not fmask[k]):
                factors[p, D, B, C] += q[i]

            elif (not fmask[i] and
                  fmask[j] and
                  not fmask[k]):
                factors[p, A, D, C] += q[j]

            elif (not fmask[i] and
                  not fmask[j] and
                  fmask[k]):
                factors[p, A, B, D] += q[k]

            elif (fmask[i] and
                  fmask[j] and
                  not fmask[k]):
                factors[p, D, D, C] += q[i] * q[j]


            elif (fmask[i] and
                  not fmask[j] and
                  fmask[k]):
                factors[p, D, B, D] += q[i] * q[k]


            elif (not fmask[i] and
                  fmask[j] and
                  fmask[k]):
                factors[p, A, D, D] += q[j] * q[k]


            elif (fmask[i] and
                  fmask[j] and
                  fmask[k]):
                factors[p, D, D, D] += q[i] * q[j] * q[k]


        return factors



    @staticmethod
    def get_full_fractions(atoms, elements):
        '''
        Return a valid fraction array for full atoms.

        atoms: Atoms object
        elements: pair of elements as a sorted list, e.g. ['Ag', 'Cu']
        '''
        fractions = [(1.0 if atom.symbol == elements[0] else 0.0)
                     for atom in atoms]
        return fractions







class FP_normalization:   # CL made by me
    
    @staticmethod
    def weight_array_by_fraction(array, fractions):        
        return array/sum(fractions) 
        
        
        