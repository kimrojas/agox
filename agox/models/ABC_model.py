import os
from abc import ABC, abstractmethod

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from agox.candidates.ABC_candidate import CandidateBaseClass
from agox.module import Module
from agox.observer import Observer
from agox.writer import Writer, agox_writer


class ModelBaseClass(Calculator, Observer, Writer, ABC):

    """Model Base Class implementation

    Attributes
    ----------
    database : AGOX Database obj
        If used for on-the-fly training on a database, this should be set
    verbose : int
        Verbosity of model logs.
    iteration_start_training : int
        When model is attached as observer it starts training after this number of
        iterations.
    update_period : int
        When model is attached as observer it updates every update_period
        iterations.
    record : set
        Training record.

    """

    def __init__(
        self,
        database=None,
        filter=None,
        order=0,
        verbose=True,
        use_counter=True,
        prefix="",
        iteration_start_training=0,
        update_period=1,
        surname="",
        gets={},
        sets={},
    ):
        """__init__ method for model base class

        If a database is supplied the model will attach itself as an observer on the database.
        If database is None the model needs to be trained manually.

        Parameters
        ----------
        database : AGOX Database obj

        order : int
            AGOX execution order
        verbose : int
            Verbosity
        use_counter : int
            Writer settings
        prefix : str
            Writer settings

        """
        Observer.__init__(self, order=order, surname=surname, gets=gets, sets=sets)
        Writer.__init__(self, verbose=verbose, use_counter=use_counter, prefix=prefix)
        Calculator.__init__(self)
        Module.__init__(self)

        self.filter = filter
        self._save_attributes = ["_ready_state"]

        self.verbose = verbose
        self.iteration_start_training = iteration_start_training
        self.update_period = update_period

        self.validation_data = []
        self._ready_state = False
        self._record = set()
        self.update = False

        self.add_observer_method(
            self.training_observer,
            gets=self.gets[0],
            sets=self.sets[0],
            order=self.order[0],
            handler_identifier="database",
        )

        if database is not None:
            self.attach_to_database(database)

    @property
    @abstractmethod
    def name(self):
        """str: Name of model. Must be implemented in child class."""
        pass

    @property
    @abstractmethod
    def implemented_properties(self):
        """:obj: `list` of :obj: `str`: Implemented properties.
        Available properties are: 'energy', 'forces', 'uncertainty'

        Must be implemented in child class.
        """
        pass

    @abstractmethod
    def predict_energy(self, atoms, **kwargs):
        """Method for energy prediction.

        Note
        ----------
        Always include **kwargs when implementing this function.

        Parameters
        ----------
        atoms : ASE Atoms obj or AGOX Candidate object
            The atoms object for which to predict the energy.

        Returns
        ----------
        float
            The energy prediction

        Must be implemented in child class.
        """
        pass

    @abstractmethod
    def train_model(self, training_data, **kwargs):
        """Method for model training.

        Note
        ----------
        Always include **kwargs when implementing this function.
        If your model is not trainable just write a method that does nothing

        Parameters
        ----------
        atoms : :obj: `list` of :obj: `ASE Atoms`
            List of ASE atoms objects or AGOX candidate objects to use as training data.
            All atoms must have a calculator with energy and other nesseary properties set, such that
            it can be accessed by .get_* methods on the atoms.


        Must be implemented in child class.

        """
        pass

    @property
    def ready_state(self):
        """bool: True if model has been trained otherwise False."""
        return self._ready_state

    @ready_state.setter
    def ready_state(self, state):
        self._ready_state = bool(state)

    def add_save_attributes(self, attribute):
        """Add attribute to save list.

        Parameters
        ----------
        attribute : str or list of str
            Name of attribute to add to save list.

        """
        if isinstance(attribute, str):
            self._save_attributes.append(attribute)
        else:
            self._save_attributes += attribute

    @agox_writer
    @Observer.observer_method
    def training_observer(self, database, state):
        """Observer method for use with on-the-fly training based data in an AGOX database.

        Note
        ----------
        This implementation simply calls the train_model method with all data in the database

        Parameters
        ----------
        atoms : AGOX Database object
            The database to keep the model trained against

        Returns
        ----------
        None

        """
        iteration = self.get_iteration_counter()

        if iteration < self.iteration_start_training:
            return
        if (
            iteration % self.update_period != 0
            and iteration != self.iteration_start_training
        ):
            return

        data = database.get_all_candidates()
        self.train_model(data)

    def add_validation_data(self, data):
        """Add validation data to model.

        Parameters
        ----------
        data : :obj: `list` of :obj: `ASE Atoms`
            List of ASE atoms objects or AGOX candidate objects to use as validation data.
            All atoms must have a calculator with energy and other nesseary properties set, such that
            it can be accessed by .get_* methods on the atoms.

        """
        if isinstance(data, list):
            self.validation_data += data
        else:
            self.validation_data.append(data)

    def predict_forces(self, atoms, **kwargs):
        """Method for forces prediction.

        The default numerical central difference force calculation method is used, but
        this can be overwritten with an analytical calculation of the force.

        Note
        ----------
        Always include **kwargs when implementing this function.

        Parameters
        ----------
        atoms : ASE Atoms obj or AGOX Candidate object
            The atoms object for which to predict the energy.

        Returns
        ----------
        np.array
            The force prediction with shape (N,3), where N is len(atoms)

        """
        return self.predict_forces_central(atoms, **kwargs)

    def predict_forces_central(
        self, atoms, acquisition_function=None, d=0.001, **kwargs
    ):
        """Numerical cenral difference forces prediction.

        Parameters
        ----------
        atoms : ASE Atoms obj or AGOX Candidate object
            The atoms object for which to predict the energy.
        acquisition_function : Acquisition function or None
            Function that takes evaluate acquisition function based on
            energy and uncertainty prediction. Used for relaxation in acquisition
            funtion if force uncertainties are not available.

        Returns
        ----------
        np.array
            The force prediction with shape (N,3), where N is len(atoms)

        """
        if acquisition_function is None:
            energy = lambda a: self.predict_energy(a)
        else:
            energy = lambda a: acquisition_function(
                *self.predict_energy_and_uncertainty(a)
            )

        e0 = self.predict_energy(atoms)
        energies = []

        for a in range(len(atoms)):
            for i in range(3):
                new_pos = atoms.get_positions()  # Try forward energy
                new_pos[a, i] += d
                atoms.set_positions(new_pos)
                if atoms.positions[a, i] != new_pos[a, i]:  # Check for constraints
                    energies.append(e0)
                else:
                    energies.append(energy(atoms))
                    atoms.positions[a, i] -= d

                new_pos = atoms.get_positions()  # Try backwards energy
                new_pos[a, i] -= d
                atoms.set_positions(new_pos)
                if atoms.positions[a, i] != new_pos[a, i]:
                    energies.append(e0)
                else:
                    energies.append(energy(atoms))
                    atoms.positions[a, i] += d

        penergies = np.array(energies[0::2])  # forward energies
        menergies = np.array(energies[1::2])  # backward energies

        forces = ((menergies - penergies) / (2 * d)).reshape(len(atoms), 3)
        return forces

    def predict_uncertainty(self, atoms, **kwargs):
        """Method for energy uncertainty prediction.

        Parameters
        ----------
        atoms : ASE Atoms obj or AGOX Candidate object
            The atoms object for which to predict the energy.

        Returns
        ----------
        float
            The energy uncertainty prediction

        """
        warning.warn("Uncertainty is not implemented and will return 0.")
        return 0

    def predict_forces_uncertainty(self, atoms, **kwargs):
        """Method for energy uncertainty prediction.

        Parameters
        ----------
        atoms : ASE Atoms obj or AGOX Candidate object
            The atoms object for which to predict the energy.

        Returns
        ----------
        np.array
            The force uncertainty prediction with shape (N,3) with N=len(atoms)

        """
        warning.warn("Uncertainty is not implemented and will return 0.")
        return np.zeros((len(atoms), 3))

    def predict_energy_and_uncertainty(self, atoms, **kwargs):
        """Method for energy and energy uncertainty prediction.

        Parameters
        ----------
        atoms : ASE Atoms obj or AGOX Candidate object
            The atoms object for which to predict the energy.

        Returns
        ----------
        float, float
            The energy and energy uncertainty prediction

        """
        return self.predict_energy(atoms, **kwargs), self.predict_uncertainty(
            atoms, **kwargs
        )

    def predict_forces_and_uncertainty(self, atoms, **kwargs):
        """Method for energy and energy uncertainty prediction.

        Parameters
        ----------
        atoms : ASE Atoms obj or AGOX Candidate object
            The atoms object for which to predict the energy.

        Returns
        ----------
        np.array, np.array
            Forces and forces uncertainty. Both with shape (N, 3) with N=len(atoms).

        """
        return self.predict_forces(atoms, **kwargs), self.predict_forces_uncertainty(
            atoms, **kwargs
        )

    def converter(self, atoms, **kwargs):
        """Converts an ASE atoms object to a format that can be used by the model

        Parameters
        ----------
        atoms : ASE Atoms obj or AGOX Candidate object
            The atoms object for which to predict the energy.

        Returns
        ----------
        object
            The converted object

        """
        return {}

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
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

        E = self.predict_energy(self.atoms)
        self.results["energy"] = E

        if "forces" in properties:
            forces = self.predict_forces(self.atoms)
            self.results["forces"] = forces

    def validate(self, **kwargs):
        """Method for validating the model.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments to pass to the validation method.

        Returns
        ----------
        dict
            Dictionary with validation results

        """
        if len(self.validation_data) == 0:
            return None

        E_true = []
        E_pred = []
        for d in self.validation_data:
            E_true.append(d.get_potential_energy())
            E_pred.append(self.predict_energy(d))

        E_true = np.array(E_true)
        E_pred = np.array(E_pred)

        return {
            "Energy MAE [eV]": np.mean(np.abs(E_true - E_pred)),
            "Energy RMSE [eV]": np.sqrt(np.mean((E_true - E_pred) ** 2)),
            "Max absolute energy error [eV]": np.max(np.abs(E_true - E_pred)),
            "Max relative energy error [%]": np.max((E_true - E_pred) / E_true)*100,
            "Min relative energy error [%]": np.min((E_true - E_pred) / E_true)*100,
        }

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
            self._record.add(d.cache_key)

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
            return data, []

        new_data = []
        old_data = []
        for d in data:
            if d.cache_key in self._record:
                old_data.append(d)
            else:
                new_data.append(d)
        return new_data, old_data

    @agox_writer
    def print_model_info(self, validation=None, **kwargs):
        """Prints model information

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to the model

        Returns
        ----------
        None
        """
        model_info = self.model_info(**kwargs)
        if validation is not None:
            model_info.append("------ Validation Info ------")
            model_info.append(
                "Validation data size: {}".format(len(self.validation_data))
            )
            for key, val in validation.items():
                model_info.append("{}: {:.3}".format(key, val))

        for s in model_info:
            self.writer(s)

    def model_info(self, **kwargs):
        """Returns model information

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to the model

        Returns
        ----------
        list of str
            The model information
        """
        return ["No model information available."]

    @agox_writer
    def save(self, path="model.h5"):
        """
        Save the model as a pickled object.

        Parameters
        ----------
        path : str, optional
            Path to save the model to. The default is 'model.h5'.
        """
        import h5py

        with h5py.File(path, "w") as f:
            for key in self._save_attributes:
                obj = self
                for k in key.split("."):
                    data = getattr(obj, k)
                    obj = data
                if data is not None:
                    f.create_dataset(key, data=data)

        self.writer("Saving model to {}".format(path))

    @agox_writer
    def load(self, path):
        """
        Load a pickle

        Parameters
        ----------
        path : str
            Path to a saved model.

        Returns
        -------
        model-object
            The loaded model object.
        """
        assert os.path.exists(path), "Path does not exist"
        import h5py

        self.writer("Loading model from {}".format(path))
        with h5py.File(path, "r") as f:
            for key in self._save_attributes:
                obj = self
                for k in key.split(".")[:-1]:
                    obj = getattr(obj, k)

                k = key.split(".")[-1]
                
                if key in f:
                    value = f[key][()]
                    setattr(obj, k, value)

        self.writer("Model loaded")

    def attach_to_database(self, database):
        from agox.databases.ABC_database import DatabaseBaseClass

        assert isinstance(database, DatabaseBaseClass)
        print(f"{self.name}: Attaching to database: {database}")
        self.attach(database)
