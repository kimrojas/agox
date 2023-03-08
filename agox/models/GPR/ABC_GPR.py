from abc import abstractmethod
from agox.models.ABC_model import ModelBaseClass
from agox.candidates import CandidateBaseClass
import numpy as np


class GPRBaseClass(ModelBaseClass):
    """
    GPR Base Class

    This class is the base class for all Gaussian Process Regression models.

    Attributes
    ----------
    descriptor : DescriptorBaseClass
        Descriptor object.
    kernel : KernelBaseClass
        Kernel object.
    prior : ModelBaseClass
        Prior model object.
    sparsifier : SparsifierBaseClass
        Sparsifier object.
    single_atom_energies : dict
        Dictionary of single atom energies.
    use_prior_in_training : bool
        Whether to use prior in training.
    centralize : bool
        Whether to centralize the energy.
    alpha : np.ndarray
        Model parameters.
    X : np.ndarray
        Training features.
    Y : np.ndarray
        Training targets.
    update : bool
        Whether to update the model.
    record : dict
        Training record.
    prior_energy : np.ndarray
        Prior energy.
    mean_energy : float
        Mean energy.
    
    Methods
    -------
    train_model(training_data, **kwargs)
        Train the model.
    predict_energy(atoms, **kwargs)
        Predict the energy of a given structure.
    predict_uncertainty(atoms, **kwargs)
        Predict the uncertainty of a given structure.
    predict_forces(atoms, return_uncertainty=False, **kwargs)
        Predict the forces of a given structure.
    predict_forces_uncertainty(atoms, **kwargs)
        Predict the uncertainty of the forces of a given structure.


    """

    def __init__(self, descriptor, kernel, prior = None, sparsifier = None,
                 single_atom_energies = None, use_prior_in_training = False,
                 centralize = False, **kwargs):

        """
        Parameters
        ----------
        descriptor : DescriptorBaseClass
            Descriptor object.
        kernel : KernelBaseClass
            Kernel object.
        prior : ModelBaseClass
            Prior model object.
        sparsifier : SparsifierBaseClass
            Sparsifier object.
        single_atom_energies : dict
            Dictionary of single atom energies.
        use_prior_in_training : bool
            Whether to use prior in training.
        centralize : bool
            Whether to centralize the energy.

        """
        super().__init__(**kwargs)

        self.descriptor = descriptor
        self.kernel = kernel
        self.prior = prior
        self.sparse = sparse
        self.sparsifier = sparsifier
        self.single_atom_energies = single_atom_energies
        self.use_prior_in_training = use_prior_in_training
        self.centralize = centralize

        # Initialize all possible model parameters
        self.alpha = None
        self.X = None
        self.Y = None

        self.update = False
        self.record = dict()
        

    @abstractmethod
    def get_features(self, atoms):
        """
        Get the features of a given structure.

        Parameters
        ----------
        atoms : Atoms
            Atoms object.

        Returns
        -------
        np.ndarray
            Features of the given structure.
        """
        pass


    @abstractmethod
    def _train_model(self):
        """
        Train the model.
        
        """
        pass

    
    def train_model(self, training_data, **kwargs):
        """
        Train the model.

        Parameters
        ----------
        training_data : list
            List of Atoms objects.

        """
        if self.update:
            self.X, self.Y = self._update(training_data, **kwargs)
        else:
            self.X, self.Y = self._preprocess(training_data)

        self._training_record(training_data)
  
        self._train_model()

        self.atoms = None
        self.ready_state = True

    
    def predict_energy(self, atoms, **kwargs):
        if self.alpha is None:
            return self.postprocess_energy(atoms, 0)
        
        x = self._features(atoms)
        k = self.kernel(self.X, x)
        e_pred = np.sum(k.T @ self.alpha)
        return self.postprocess_energy(atoms, e_pred)

    
    def predict_uncertainty(self, atoms, **kwargs):
        self.writer('Uncertainty not implemented.')
        return 0

    
    def predict_forces(self, atoms, return_uncertainty=False, **kwargs):
        f_pred = self.predict_forces_central(atoms, **kwargs)

        if return_uncertainty:
            return self.postprocess_forces(atoms, f_pred), self.predict_forces_uncertainty(atoms, **kwargs)
        else:
            return self.postprocess_forces(atoms, f_pred)

        
    def predict_forces_uncertainty(self, atoms, **kwargs):
        self.writer('Uncertainty not implemented.')
        return np.zeros((len(atoms), 3))

    
    @property
    def single_atom_energies(self):
        """
        Get the single atom energies.

        Returns
        -------
        np.ndarray

        """
        return self._single_atom_energies

    
    @single_atom_energies.setter
    def single_atom_energies(self, s):
        """
        Set the single atom energies.
        Index number corresponds to the atomic number ie. 1 = H, 2 = He, etc.

        Parameters
        ----------
        s : dict or np.ndarray
            Dictionary/array of single atom energies.

        """
        if isinstance(s, np.ndarray):
            self._single_atom_energies = s
        elif isinstance(s, dict):
            self._single_atom_energies = np.zeros(100)
            for i, val in s.items():
                self._single_atom_energies[i] = val
        elif s is None:
            self._single_atom_energies = np.zeros(100)


    def _preprocess(self, data):
        """
        Preprocess the training data.

        Parameters
        ----------
        data : list
            List of Atoms objects.

        Returns
        -------
        np.ndarray
            Features.
        np.ndarray
            Targets.
        
        """
        Y = np.expand_dims(np.array([d.get_potential_energy() for d in data]), axis=1)
        
        if self.prior is None:
            self.prior_energy = np.zeros(Y.shape)
        else:
            self.prior_energy = np.expand_dims(np.array([self.prior.predict_energy(d) for d in data]), axis=1)

        Y -= self.prior_energy
        
        if self.centralize:
            self.mean_energy = np.mean(Y)
        else:
            self.mean_energy = 0.
            
        Y -= self.mean_energy
        X = self.get_features(data)

        return X, Y


    def _update(self, training_data):
        """
        Update the the features and targets.
        training_data is all the data.

        Parameters
        ----------
        training_data : list
            List of Atoms objects.

        Returns
        -------
        np.ndarray
            Features.
        np.ndarray
            Targets.
        
        """
        return self._preprocess(training_data)

        
    def postprocess_energy(self, atoms, e_pred):
        """
        Postprocess the energy.

        Parameters
        ----------
        atoms : Atoms
            Atoms object.
        e_pred : float
            Predicted energy.

        Returns
        -------
        float
            Postprocessed energy.
        """
        if self.prior is None:
            prior = 0
        else:
            prior = self.prior.predict_energy(atoms)
            
        return e_pred + prior + self.mean_energy

    
    def postprocess_forces(self, atoms, f_pred):
        """
        Postprocess the forces.

        Parameters
        ----------
        atoms : Atoms
            Atoms object.
        f_pred : np.ndarray
            Predicted forces.

        Returns
        -------
        np.ndarray
            Postprocessed forces.
        """
        if self.prior is None:
            prior = 0
        else:
            prior = self.prior.predict_forces(atoms)
            
        return f_pred + prior
            

    def _training_record(self, data):
        """
        Record the training data.

        Parameters
        ----------
        data : list
            List of Atoms objects.

        """
        if not all([isinstance(d, CandidateBaseClass) for d in data]):
            return

        for d in data:
            self.record[d.cache_key] = True

        self.update = True

            
    def _get_new_data(self, data):
        """
        Get the new data.

        Parameters
        ----------
        data : list
            List of Atoms objects.

        Returns
        -------
        list
            List of new Atoms objects.

        list
            List of old Atoms objects.
        
        """
        if not all([isinstance(d, CandidateBaseClass) for d in data]):
            return data

        new_data = []
        old_data = []
        for d in data:
            if d.cache_key in self.record:
                old_data.append(d)
            else:
                new_data.append(d)
        return new_data, old_data


        
