from abc import ABC, abstractmethod

class SparsifierBaseClass(ABC):
    """
    Abstract class for sparsifiers.

    A sparsifier is a function that takes a matrix X with rows corresponding to feaures.
    It returns a matrix with the same number of columns, possibly together with which rows are selected.


    Attributes
    ----------
    m_points : int
        Number of points to select

    Methods
    -------
    sparsify(X)
        Returns a matrix with the same number of columns together with which rows are selected (if available).

    """

    def __init__(self, m_points=1000, **kwargs):
        """
        Parameters
        ----------
        m_points : int
            Number of points to select
        """
        self.m_points = m_points
    
    @abstractmethod
    def sparsify(self, X):
        """
        Returns a matrix with the same number of columns together with which rows are selected.

        Parameters
        ----------
        X : array-like, shape (n_features, n_samples)
            Matrix with rows corresponding to feaures.

        Returns
        -------
        X_sparsified : array-like, shape (m_points, n_samples)
            Matrix with rows corresponding to feaures.
        selected_rows : array-like, shape (m_points,)
            Indices of selected rows.

        """
        pass

    def __call__(self, X):
        return self.sparsify(X)
