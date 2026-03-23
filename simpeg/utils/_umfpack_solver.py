"""UMFPACK direct solver for use as a pymatsolver backend.

Uses SuiteSparse's UMFPACK via scikit-umfpack. This provides a fast direct
solver that works on ARM/Apple Silicon, where Intel MKL's Pardiso is not
available.
"""

import numpy as np
import scipy.sparse as sp
from pymatsolver.solvers import Base

try:
    from scikits.umfpack import UmfpackContext, UMFPACK_A, UMFPACK_At

    _umfpack_available = True
except ImportError:
    _umfpack_available = False

__all__ = ["SolverUMFPACK", "is_available"]


def is_available():
    """Return whether the UMFPACK solver is available.

    Returns
    -------
    bool
        ``True`` if scikit-umfpack is installed and importable.
    """
    return _umfpack_available


class SolverUMFPACK(Base):
    """Direct solver using SuiteSparse UMFPACK.

    UMFPACK is a set of routines for solving unsymmetric sparse linear
    systems using the Unsymmetric MultiFrontal method. It is part of
    SuiteSparse and is available on all platforms including ARM/Apple
    Silicon.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Matrix to solve with. Will be converted to CSC format if needed.
    check_accuracy : bool, optional
        Whether to check the accuracy of the solution.
    check_rtol : float, optional
        The relative tolerance to check against for accuracy.
    check_atol : float, optional
        The absolute tolerance to check against for accuracy.
    **kwargs
        Extra keyword arguments passed to the base class.

    Raises
    ------
    ImportError
        If scikit-umfpack is not installed.
    """

    _transposed = False

    def __init__(
        self,
        A,
        check_accuracy=False,
        check_rtol=1e-6,
        check_atol=0,
        **kwargs,
    ):
        if not _umfpack_available:
            raise ImportError(
                "UMFPACK solver requires the scikit-umfpack package. "
                "Install it with: pip install scikit-umfpack"
            )
        if not (sp.issparse(A) and A.format == "csc"):
            A = sp.csc_matrix(A)
        A.sum_duplicates()
        super().__init__(
            A,
            check_accuracy=check_accuracy,
            check_rtol=check_rtol,
            check_atol=check_atol,
            **kwargs,
        )
        family = "zi" if np.issubdtype(A.dtype, np.complexfloating) else "di"
        self._umfpack = UmfpackContext(family)
        self._umfpack.numeric(self.A)

    def factor(self, A=None):
        """(Re)factor the A matrix.

        Parameters
        ----------
        A : scipy.sparse.spmatrix, optional
            New matrix to factorize. If ``None``, the existing matrix is used.
        """
        if A is not None and self.A is not A:
            if not (sp.issparse(A) and A.format == "csc"):
                A = sp.csc_matrix(A)
            A.sum_duplicates()
            self._A = A
            family = "zi" if np.issubdtype(A.dtype, np.complexfloating) else "di"
            self._umfpack = UmfpackContext(family)
        self._umfpack.numeric(self.A)

    def _solve_single(self, rhs):
        sys = UMFPACK_At if self._transposed else UMFPACK_A
        return self._umfpack.solve(sys, self.A, rhs, autoTranspose=True)

    def _solve_multiple(self, rhs):
        sys = UMFPACK_At if self._transposed else UMFPACK_A
        X = np.empty_like(rhs)
        for i in range(rhs.shape[1]):
            X[:, i] = self._umfpack.solve(sys, self.A, rhs[:, i], autoTranspose=True)
        return X

    def transpose(self):
        """Return a solver that solves the transposed system."""
        trans_obj = SolverUMFPACK.__new__(SolverUMFPACK)
        trans_obj._A = self.A
        for attr, value in self.get_attributes().items():
            setattr(trans_obj, attr, value)
        trans_obj._umfpack = self._umfpack
        trans_obj._transposed = not self._transposed
        # Keep a reference to prevent GC from freeing the shared context
        # when the original solver goes out of scope.
        trans_obj._parent = self
        return trans_obj

    def clean(self):
        """Free the UMFPACK numeric factorization."""
        if hasattr(self, "_umfpack"):
            self._umfpack.free()
