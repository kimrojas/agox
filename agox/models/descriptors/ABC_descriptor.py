import numpy as np
from abc import ABC, abstractmethod
from agox.module import Module
from agox.candidates.ABC_candidate import CandidateBaseClass

all_feature_types = ['global', 'local', 'global_gradient', 'local_gradient']

class DescriptorBaseClass(ABC, Module):

    feature_types = []

    def __init__(self, surname='', use_cache=False, **kwargs):
        Module.__init__(self, surname=surname)
        assert np.array([feature_type in all_feature_types for feature_type in self.feature_types]).all(), 'Unknown feature type declared.'
        self.use_cache = use_cache
        self._cache_key = self.name + '/' + str(hash(self))


    ##########################################################################################################
    # Create methods - Implemented by classes that inherit from this base-class.
    # Not called directly by methods of other classes. 
    ##########################################################################################################

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

    def create_global_feature_gradient(self, atoms):
        """
        Method to implement on child classes that does the calculate of 
        global feature gradients. 

        This method should not (generally) deal with being given a list of 
        Atoms objects. 

        If implemented 'global_gradient' can be added to the child class' feature_types.

        Parameters
        ----------
        atoms : Atoms object

        Returns
        --------
        A single global feature gradient.
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

    def create_local_feature_gradient(self, atoms):
        """
        Method to implement on child classes that does the calculate of 
        local feature gradients.

        This method should not (generally) deal with being given a list of 
        Atoms objects. 

        If implemented 'local_gradient' can be added to the child class' feature_types.

        Parameters
        ----------
        atoms : Atoms object

        Returns
        --------
        A single global feature gradient.
        """
        pass

    ##########################################################################################################
    # Get methods - Ones to use in other scripts.
    ##########################################################################################################

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
        feature_type = 'global'
        self.feature_type_check(feature_type)
        if not (type(atoms) == list):
            atoms = [atoms]
            
        features = []
        for a in atoms:
            f = self.get_from_cache(a, feature_type)
            if f is None:
                f = self.create_global_features(a)
                self.set_to_cache(a, f, feature_type)
                
            features.append(f)
            
        return features

    def get_global_feature_gradient(self, atoms):
        """
        Method to get global features.

        Parameters
        ----------
        atoms : Atoms object or list or Atoms objects.
            Atoms to calculate features for. 

        Returns
        -------
        list
            Global feature gradients for the given atoms.
        """
        feature_type = 'global_gradient'
        self.feature_type_check(feature_type)
        if not (type(atoms) == list):
            atoms = [atoms]

        features = []
        for a in atoms:
            f = self.get_from_cache(a, feature_type)
            if f is None:
                f = self.create_global_feature_gradient(a)
                self.set_to_cache(a, f, feature_type)
            features.append(f)
            
        return features

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
        feature_type = 'local'
        self.feature_type_check(feature_type)
        if not (type(atoms) == list):
            atoms = [atoms]

        features = []
        for a in atoms:
            f = self.get_from_cache(a, feature_type)
            if f is None:
                f = self.create_local_features(a)
                self.set_to_cache(a, f, feature_type)
            features.append(f)
            
        return features

    def get_local_feature_gradient(self, atoms):
        """
        Method to get local feature gradients.

        Parameters
        ----------
        atoms : Atoms object or list or Atoms objects.
            Atoms to calculate features for. 

        Returns
        -------
        list
            Local feature gradients for the given atoms.
        """
        feature_type = 'local_gradient'
        self.feature_type_check(feature_type)
        if not (type(atoms) == list):
            atoms = [atoms]
            
        features = []
        for a in atoms:
            f = self.get_from_cache(a, feature_type)
            if f is None:
                f = self.create_local_feature_gradient(a)
                self.set_to_cache(a, f, feature_type)
            features.append(f)
            
        return features

    def feature_type_check(self, feature_type):
        if not feature_type in self.feature_types:
            raise NotImplementedError(f'This descriptor does not support {feature_type} features')

    def get_from_cache(self, atoms, feature_type):
        if not self.use_cache or not isinstance(atoms, CandidateBaseClass):
            return None
        
        return atoms.get_from_cache(feature_type + '/' + self._cache_key)
            
    def set_to_cache(self, atoms, value, feature_type):
        if self.use_cache and isinstance(atoms, CandidateBaseClass):
            atoms.cache(feature_type + '/' + self._cache_key, value)

