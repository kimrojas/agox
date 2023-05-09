"""
Models are used to predict the output of a given input. Generally this corresponds 
to predicting the energy and forces of a given atomic configuration. 

This module also contains the descriptors used to represent the atomic configurations.
"""
from .ABC_model import ModelBaseClass
from .ABC_model import load
from .GPR import ModelGPR

