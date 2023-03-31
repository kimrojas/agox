import warnings
from typing import Dict, List, Tuple

import numpy as np
from ase import Atoms
from scipy.linalg import lstsq, qr

from agox.models.GPR.GPR import GPR
from agox.utils import candidate_list_comprehension
from agox.utils.sparsifiers import CUR, SparsifierBaseClass


class SparseGPR(GPR):
    name = "SparseGPR"

    implemented_properties = ["energy", "forces", "local_energy"]

    dynamic_attributes = ["Xm", "K_inv", "Kmm_inv", "alpha"]

    """
    Sparse GPR Class

    Attributes
    ----------
    Xn : np.ndarray
        Training data
    Xm : np.ndarray
        Inducing points
    K_mm : np.ndarray
        Kernel matrix between inducing points
    K_nm : np.ndarray
        Kernel matrix between training and inducing points
    K_inv : np.ndarray
        Inverse of K_mm + K_nm.T @ sigma_inv @ K_nm
    Kmm_inv : np.ndarray
        Inverse of K_mm
    L : np.ndarray
        Matrix of ones and zeros indicating which atoms are in which configuration
    sparsifier: SparsifierBaseClass
        Sparsifier object
    

    Methods
    -------
    predict_local_energy(atoms=None, X=None)
        Calculate the local energies in the model.
    
    """

    def __init__(
        self,
        noise: float = 0.05,
        centralize: bool = False,
        jitter: float = 1e-8,
        filter=None,
        sparsifier: SparsifierBaseClass = CUR(),
        **kwargs
    ) -> None:
        """

        Parameters
        ----------
        noise : float
            Noise level
        sparsifier : SparsifierBaseClass
            Sparsifier object
        jitter : float
            Jitter level

        """

        super().__init__(centralize=centralize, filter=filter**kwargs)

        self.jitter = jitter
        self.noise = noise

        self.sparsifier = sparsifier

        self._transfer_data = []
        self._transfer_noise = np.array([])

        self.add_save_attributes(["Xn", "Xm", "K_mm", "K_nm", "Kmm_inv", "L"])
        self.Xn = None
        self.Xm = None
        self.K_mm = None
        self.K_nm = None
        self.Kmm_inv = None
        self.L = None

    @candidate_list_comprehension
    def predict_forces(self, atoms: Atoms, **kwargs) -> np.ndarray:
        """Method for forces prediction.

        Parameters
        ----------
        atoms : ase.Atoms
            ase.Atoms object to predict forces for

        Returns
        ----------
        np.array
            The force prediction with shape (N,3), where N is len(atoms)

        """
        return self.predict_forces_central(atoms, **kwargs)

    @candidate_list_comprehension
    def predict_local_energy(self, atoms: Atoms, **kwargs) -> np.ndarray:
        """
        Calculate the local energies in the model.

        Parameters
        ----------
        atoms : ase.Atoms
            ase.Atoms object
        X : np.ndarray
            Features for the ase.Atoms object

        Returns
        -------
        np.ndarray
            Local energies

        """
        X = self._get_features(atoms)
        k = self.kernel(self.Xm, X)
        return (k.T @ self.alpha).reshape(
            -1,
        ) + self.single_atom_energies[atoms.get_atomic_numbers()]

    @property
    def noise(self) -> float:
        """
        Noise level

        Returns
        -------
        float
            Noise level

        """
        return self._noise

    @noise.setter
    def noise(self, s: float) -> None:
        """
        Noise level

        Parameters
        ----------
        s : float
            Noise level

        """
        self._noise = s

    @property
    def transfer_data(self) -> List[Atoms]:
        """
        List of ase.Atoms objects to transfer to the model

        Returns
        -------
        list of ase.Atoms
            List of ase.Atoms objects to transfer to the model

        """
        return self._transfer_data

    @transfer_data.setter
    def transfer_data(self, l: List[Atoms]) -> None:
        """
        List of ase.Atoms objects to transfer to the model

        Parameters
        ----------
        l : list of ase.Atoms
            ase.Atoms objects to transfer to the model

        """
        warnings.warn(
            "transfer_data should not be used. Use add_transfer_data instead."
        )
        self.add_transfer_data(l)

    @property
    def transfer_noise(self) -> np.ndarray:
        """
        Noise level for transfer data

        Returns
        -------
        np.ndarray
            Noise level for transfer data

        """
        return self._transfer_noise

    @transfer_noise.setter
    def transfer_noise(self, s: np.ndarray) -> None:
        """
        Noise level for transfer data

        Parameters
        ----------
        s : np.ndarray
            Noise level for transfer data

        """
        warnings.warn(
            "transfer_noise should not be used. Use add_transfer_data instead."
        )
        self._transfer_noise = s

    def add_transfer_data(self, data: List[Atoms], noise: float = None) -> None:
        """
        Add ase.Atoms objects to the transfer data

        Parameters
        ----------
        data : list of ase.Atoms
            List of ase.Atoms objects to add to the transfer data
        noise : float
            Noise level for the transfer data

        """
        if isinstance(data, list):
            self._transfer_data += data
            if noise is None:
                noise = self.noise

            self._transfer_noise = np.append(
                self._transfer_noise, np.ones(len(data)) * noise
            )

        else:
            self.add_transfer_data.append([data])

    def model_info(self, **kwargs) -> List[str]:
        """
        List of strings with model information
        """
        x = "    "
        sparsifier_name = (
            self.sparsifier.name if self.sparsifier is not None else "None"
        )
        sparsifier_mpoints = (
            self.sparsifier.m_points if self.sparsifier is not None else "None"
        )
        out = [
            "------ Model Info ------",
            "Descriptor:",
            x + "{}".format(self.descriptor.name),
            "Kernel:",
            x + "{}".format(self.kernel),
            "Sparsifier:",
            x + "{} selecting {} points".format(sparsifier_name, sparsifier_mpoints),
            "Noise:",
            x + "{}".format(self.noise),
            "------ Training Info ------",
            "Training data size: {}".format(self.X.shape[0]),
            "Number of local environments: {}".format(self.Xn.shape[0]),
            "Number of inducing points: {}".format(self.Xm.shape[0]),
        ]

        return out

    def _train_model(self, data: List[Atoms]) -> None:
        """
        Train the model

        """
        assert self.Xn is not None, "self.Xn must be set prior to call"
        assert self.L is not None, "self.L must be set prior to call"
        assert self.Y is not None, "self.Y must be set prior to call"

        self.Xm, _ = self.sparsifier(atoms=self.transfer_data + data, X=self.Xn)
        self.X = self.Xm

        self.K_mm = self.kernel(self.Xm)
        self.K_nm = self.kernel(self.Xn, self.Xm)

        LK_nm = self.L @ self.K_nm
        K = (
            self.K_mm
            + LK_nm.T @ self.sigma_inv @ LK_nm
            + self.jitter * np.eye(self.K_mm.shape[0])
        )

        K = self._symmetrize(K)
        Q, R = qr(K)
        self.K_inv = lstsq(R, Q.T)[0]
        self.alpha = lstsq(R, Q.T @ LK_nm.T @ self.sigma_inv @ self.Y)[0]

    def _preprocess(self, data: List[Atoms]) -> None:
        """
        Preprocess the training data for the model

        Parameters
        ----------
        data : list of ase.Atoms
            List of ase.Atoms objects

        Returns
        -------
        np.ndarray
            Features for the ase.Atoms objects
        np.ndarray
            Energies for the ase.Atoms objects

        """
        X, Y = super()._preprocess(self.transfer_data + data)
        self.Xn = X

        self.L = self._make_L(self.transfer_data + data, X.shape)
        self.sigma_inv = self._make_sigma(self.transfer_data + data)

        return X, Y

    def _update(self, data: List[Atoms]) -> None:
        """
        Update the training data for the model

        Parameters
        ----------
        data : list of ase.Atoms
            List of ase.Atoms objects

        Returns
        -------
        np.ndarray
            Features for the ase.Atoms objects
        np.ndarray
            Energies for the ase.Atoms objects

        """
        new, old = self._get_new_data(data)
        if len(new) == len(data):
            return self._preprocess(data)

        X_new, Y_new = super()._preprocess(new)

        X = np.vstack((self.X, X_new))
        self.Xn = X

        Y = np.vstack((self.Y, Y_new))

        self.L = self._update_L(new, X_new.shape)
        self.sigma_inv = self._make_sigma(self.transfer_data + data)

        return X, Y

    def _make_L(self, atoms_list: List[Atoms], shape_X: Tuple[int, int]) -> np.ndarray:
        """
        Make the L matrix

        Parameters
        ----------
        atoms_list : list of ase.Atoms
            List of ase.Atoms objects

        Returns
        -------
        np.ndarray
            L matrix

        """
        if len(atoms_list) == shape_X[0]:
            return np.eye(shape_X[0])

        lengths = [len(atoms) for atoms in atoms_list]
        r = len(lengths)
        c = np.sum(lengths)

        col = 0
        L = np.zeros((r, c))
        for i, atoms in enumerate(atoms_list):
            L[i, col : col + len(atoms)] = 1.0
            col += len(atoms)
        return L

    def _update_L(
        self, new_atoms_list: List[Atoms], shape_X: Tuple[int, int]
    ) -> np.ndarray:
        """
        Update the L matrix

        Parameters
        ----------
        new_atoms_list : list of ase.Atoms
            List of ase.Atoms objects

        Returns
        -------
        np.ndarray
            L matrix

        """
        if len(new_atoms_list) == shape_X[0]:  # this probably doesn't work
            new_size = shape_X[0] + self.L.shape[0]
            return np.eye(new_size)

        new_lengths = [len(atoms) for atoms in new_atoms_list]
        size = len(new_lengths)
        new_total_length = np.sum(new_lengths)
        new_L = np.zeros((self.L.shape[0] + size, self.L.shape[1] + new_total_length))
        new_L[0 : self.L.shape[0], 0 : self.L.shape[1]] = self.L

        for l in range(size):
            step = int(np.sum(new_lengths[:l]))
            new_L[
                l + self.L.shape[0],
                (self.L.shape[1] + step) : (self.L.shape[1] + step + new_lengths[l]),
            ] = 1
        return new_L

    def _make_sigma(self, atoms_list: List[Atoms]) -> np.ndarray:
        """
        Make the sigma matrix

        Parameters
        ----------
        atoms_list : list of ase.Atoms
            List of ase.Atoms objects

        Returns
        -------
        np.ndarray
            Sigma matrix

        """
        sigmas = np.array([self.noise**2 * len(atoms) for atoms in atoms_list])
        sigmas[: len(self.transfer_data)] = self.transfer_noise**2 * np.array(
            [len(atoms) for atoms in self.transfer_data]
        )

        sigma_inv = np.diag(1 / sigmas)
        return sigma_inv

    def _symmetrize(self, A: np.ndarray) -> np.ndarray:
        """
        Symmetrize a matrix

        Parameters
        ----------
        A : np.ndarray
            Matrix to symmetrize

        Returns
        -------
        np.ndarray
            Symmetrized matrix

        """
        return (A + A.T) / 2
