#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:51:38 2023

@author: casper
"""

from agox.models.descriptors.ABC_descriptor import DescriptorBaseClass
from gpatom.gpfp.fingerprint import  RadialAngularFP, RadialFP, CartesianCoordFP

#from math import pi
#import itertools
#from scipy.spatial import distance_matrix
#import numpy as np
#import ase


class BeaconFingerPrint(DescriptorBaseClass):
    
    feature_types = ['global', 'global_gradient']    
    
    def __init__(self, fp_args={}, weight_by_elements=False, calc_gradients=True, **kwargs):
        super().__init__(**kwargs)
        self.fp_args=fp_args
        self.weight_by_elements=weight_by_elements
        self.calc_gradients=calc_gradients
        
        
    def get_fp_object(self, atoms):    
        
        fp_object= RadialAngularFP(atoms, 
                            weight_by_elements=self.weight_by_elements,
                            calc_gradients=self.calc_gradients,
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

  