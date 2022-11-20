import numpy as np
from abc import ABC, abstractmethod
#from agox.module import Module

all_feature_types = ['global', 'local', 'global_derivative', 'local_derivative']

#class DescriptorBaseClass(ABC, Module):
class DescriptorBaseClass(ABC):

    feature_types = []

    def __init__(self, surname='', **kwargs):
        #Module.__init__(self, surname=surname)
        assert np.array([feature_type in all_feature_types for feature_type in self.feature_types]).all(), 'Unknown feature type declared.'

    def create_global_features(self, atoms):
        """
        Method to implement on child classes that does the calculate of 
        global feature vectors. 

        This method should not (generally) deal with being given a list of 
        Atoms objects. 

        If implemented 'global' can be added to the child class' feature_types.

        Parameters
        ----------
        atoms : Atoms object

        Returns
        --------
        A single global feature.
        """
        pass

    def create_global_feature_derivatives(self, atoms):
        """
        Method to implement on child classes that does the calculate of 
        global feature derivatives. 

        This method should not (generally) deal with being given a list of 
        Atoms objects. 

        If implemented 'global_derivative' can be added to the child class' feature_types.

        Parameters
        ----------
        atoms : Atoms object

        Returns
        --------
        A single global feature derivative.
        """
        pass

    def create_local_features(self, atoms):
        """
        Method to implement on child classes that does the calculate of 
        local features.

        This method should not (generally) deal with being given a list of 
        Atoms objects. 

        If implemented 'local' can be added to the child class' feature_types.

        Parameters
        ----------
        atoms : Atoms object

        Returns
        --------
        Local features for each atom in Atoms.
        """
        pass

    def create_local_feature_derivatives(self, atoms):
        """
        Method to implement on child classes that does the calculate of 
        local feature derivatives.

        This method should not (generally) deal with being given a list of 
        Atoms objects. 

        If implemented 'local_derivative' can be added to the child class' feature_types.

        Parameters
        ----------
        atoms : Atoms object

        Returns
        --------
        A single global feature derivative.
        """
        pass

    def get_global_features(self, atoms):
        """
        Method to get global features.

        Parameters
        ----------
        atoms : Atoms object or list or Atoms objects.
            Atoms to calculate features for. 

        Returns
        -------
        list
            Global features for the given atoms.
        """
        self.feature_type_check('global')
        if not (type(atoms) == list):
            atoms = [atoms]
        return [self.create_global_features(a) for a in atoms]

    def get_global_feature_derivatives(self, atoms):
        """
        Method to get global features.

        Parameters
        ----------
        atoms : Atoms object or list or Atoms objects.
            Atoms to calculate features for. 

        Returns
        -------
        list
            Global feature derivatives for the given atoms.
        """
        self.feature_type_check('global_derivative')
        if not (type(atoms) == list):
            atoms = [atoms]
        return [self.create_global_feature_derivatives(a) for a in atoms]

    def get_local_features(self, atoms):
        """
        Method to get local features.

        Parameters
        ----------
        atoms : Atoms object or list or Atoms objects.
            Atoms to calculate features for. 

        Returns
        -------
        list
            Local features for the given atoms.
        """
        self.feature_type_check('local')
        if not (type(atoms) == list):
            atoms = [atoms]
        return [self.create_local_features(a) for a in atoms]

    def get_local_feature_derivatives(self, atoms):
        """
        Method to get local feature derivatives.

        Parameters
        ----------
        atoms : Atoms object or list or Atoms objects.
            Atoms to calculate features for. 

        Returns
        -------
        list
            Local feature derivatives for the given atoms.
        """
        self.feature_type_check('local_derivative')
        if not (type(atoms) == list):
            atoms = [atoms]
        return [self.create_local_feature_derivatives(a) for a in atoms]

    def feature_type_check(self, feature_type):
        if not feature_type in self.feature_types:
            raise NotImplementedError(f'This descriptor does not support {feature_type} features')