"""Tests for the UMFPACK solver wrapper."""

import numpy as np
import scipy.sparse as sp
import pytest
from pymatsolver import SolverLU
from pymatsolver.solvers import Base

from simpeg.utils._umfpack_solver import SolverUMFPACK, is_available

# Skip the entire module if scikit-umfpack is not installed
pytestmark = pytest.mark.skipif(
    not is_available(), reason="scikit-umfpack is not installed"
)


@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def spd_system(rng):
    """A symmetric positive definite sparse system."""
    n = 200
    A = sp.random(n, n, density=0.05, format="csc", random_state=rng)
    A = A @ A.T + 5.0 * sp.eye(n, format="csc")
    b = rng.standard_normal(n)
    return A, b


@pytest.fixture()
def unsymmetric_system(rng):
    """An unsymmetric sparse system."""
    n = 200
    A = sp.random(n, n, density=0.05, format="csc", random_state=rng)
    A = A + 10.0 * sp.eye(n, format="csc")
    b = rng.standard_normal(n)
    return A, b


@pytest.fixture()
def poisson_3d():
    """A 3D Poisson system (15^3 = 3375 unknowns)."""
    n = 15
    N = n**3
    e = np.ones(N)
    diags = [-6 * e, e, e, e, e, e, e]
    offsets = [0, 1, -1, n, -n, n * n, -(n * n)]
    A = sp.diags(diags, offsets, shape=(N, N), format="csc")
    A = A + 0.01 * sp.eye(N, format="csc")
    rng = np.random.default_rng(42)
    b = rng.standard_normal(N)
    return A, b


class TestSolverUMFPACK:
    """Tests for the SolverUMFPACK class."""

    def test_is_subclass_of_base(self):
        assert issubclass(SolverUMFPACK, Base)

    def test_solve_unsymmetric(self, unsymmetric_system):
        A, b = unsymmetric_system
        solver = SolverUMFPACK(A)
        x = solver * b
        np.testing.assert_allclose(A @ x, b, rtol=1e-10)

    def test_solve_spd(self, spd_system):
        A, b = spd_system
        solver = SolverUMFPACK(A)
        x = solver * b
        np.testing.assert_allclose(A @ x, b, rtol=1e-10)

    def test_solve_poisson_3d(self, poisson_3d):
        A, b = poisson_3d
        solver = SolverUMFPACK(A)
        x = solver * b
        np.testing.assert_allclose(A @ x, b, rtol=1e-10)

    def test_solve_multiple_rhs(self, unsymmetric_system, rng):
        A, _ = unsymmetric_system
        n = A.shape[0]
        B = rng.standard_normal((n, 5))
        solver = SolverUMFPACK(A)
        X = solver * B
        np.testing.assert_allclose(A @ X, B, rtol=1e-10)

    def test_transpose_solve(self, unsymmetric_system):
        A, b = unsymmetric_system
        solver = SolverUMFPACK(A)
        solver_T = solver.T
        x = solver_T * b
        np.testing.assert_allclose(A.T @ x, b, rtol=1e-10)

    def test_csr_input_converted(self, unsymmetric_system):
        """Solver should accept CSR matrices by converting to CSC."""
        A, b = unsymmetric_system
        A_csr = A.tocsr()
        solver = SolverUMFPACK(A_csr)
        x = solver * b
        np.testing.assert_allclose(A @ x, b, rtol=1e-10)

    def test_refactor(self, rng):
        """Test re-factorization with a new matrix."""
        n = 100
        A1 = sp.random(n, n, density=0.1, format="csc", random_state=rng) + 10 * sp.eye(
            n
        )
        A2 = sp.random(n, n, density=0.1, format="csc", random_state=rng) + 10 * sp.eye(
            n
        )
        b = rng.standard_normal(n)

        solver = SolverUMFPACK(A1)
        x1 = solver * b
        np.testing.assert_allclose(A1 @ x1, b, rtol=1e-10)

        solver.factor(A2)
        x2 = solver * b
        np.testing.assert_allclose(A2 @ x2, b, rtol=1e-10)

    def test_complex_system(self, rng):
        """Test with a complex-valued matrix."""
        n = 100
        A = (
            sp.random(n, n, density=0.1, format="csc", random_state=rng)
            + 1j * sp.random(n, n, density=0.1, format="csc", random_state=rng)
            + 10 * sp.eye(n)
        )
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        solver = SolverUMFPACK(A)
        x = solver * b
        np.testing.assert_allclose(A @ x, b, rtol=1e-10)

    def test_clean(self, unsymmetric_system):
        A, b = unsymmetric_system
        solver = SolverUMFPACK(A)
        solver.clean()


class TestUMFPACKParityWithSolverLU:
    """Verify UMFPACK and SolverLU (SuperLU) produce identical results."""

    def test_unsymmetric(self, unsymmetric_system):
        A, b = unsymmetric_system
        x_umf = SolverUMFPACK(A) * b
        x_lu = SolverLU(A) * b
        np.testing.assert_allclose(x_umf, x_lu, rtol=1e-10)

    def test_spd(self, spd_system):
        A, b = spd_system
        x_umf = SolverUMFPACK(A) * b
        x_lu = SolverLU(A) * b
        np.testing.assert_allclose(x_umf, x_lu, rtol=1e-10)

    def test_poisson_3d(self, poisson_3d):
        A, b = poisson_3d
        x_umf = SolverUMFPACK(A) * b
        x_lu = SolverLU(A) * b
        np.testing.assert_allclose(x_umf, x_lu, rtol=1e-10)

    def test_multiple_rhs(self, unsymmetric_system, rng):
        A, _ = unsymmetric_system
        B = rng.standard_normal((A.shape[0], 5))
        X_umf = SolverUMFPACK(A) * B
        X_lu = SolverLU(A) * B
        np.testing.assert_allclose(X_umf, X_lu, rtol=1e-10)

    def test_transpose(self, unsymmetric_system):
        A, b = unsymmetric_system
        x_umf = SolverUMFPACK(A).T * b
        x_lu = SolverLU(A).T * b
        np.testing.assert_allclose(x_umf, x_lu, rtol=1e-10)

    def test_complex(self, rng):
        n = 100
        A = (
            sp.random(n, n, density=0.1, format="csc", random_state=rng)
            + 1j * sp.random(n, n, density=0.1, format="csc", random_state=rng)
            + 10 * sp.eye(n)
        )
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x_umf = SolverUMFPACK(A) * b
        x_lu = SolverLU(A) * b
        np.testing.assert_allclose(x_umf, x_lu, rtol=1e-10)

    def test_ill_conditioned(self, rng):
        """Both solvers should agree even on poorly conditioned systems."""
        n = 50
        A = sp.random(n, n, density=0.2, format="csc", random_state=rng)
        A = A + 0.01 * sp.eye(n, format="csc")  # barely nonsingular
        b = rng.standard_normal(n)
        x_umf = SolverUMFPACK(A) * b
        x_lu = SolverLU(A) * b
        # Looser tolerance for ill-conditioned system
        np.testing.assert_allclose(x_umf, x_lu, rtol=1e-6)

    def test_csr_input(self, unsymmetric_system):
        """Both solvers should handle CSR input identically."""
        A, b = unsymmetric_system
        A_csr = A.tocsr()
        x_umf = SolverUMFPACK(A_csr) * b
        x_lu = SolverLU(A_csr) * b
        np.testing.assert_allclose(x_umf, x_lu, rtol=1e-10)


class TestUMFPACKDefaultSolver:
    """Test that UMFPACK is picked up in the default solver chain."""

    def test_is_available(self):
        assert is_available() is True

    def test_default_solver_prefers_umfpack_over_superlu(self):
        """On a system without Pardiso/Mumps, UMFPACK should be default."""
        from pymatsolver import AvailableSolvers
        from simpeg.utils import get_default_solver

        default = get_default_solver()
        if AvailableSolvers["Pardiso"]:
            pytest.skip("Pardiso is available, it takes priority")
        elif AvailableSolvers["Mumps"]:
            pytest.skip("Mumps is available, it takes priority")
        else:
            assert (
                default is SolverUMFPACK
            ), f"Expected SolverUMFPACK as default, got {default}"

    def test_set_default_solver_accepts_umfpack(self):
        from simpeg.utils import get_default_solver, set_default_solver

        initial = get_default_solver()
        try:
            set_default_solver(SolverUMFPACK)
            assert get_default_solver() is SolverUMFPACK
        finally:
            set_default_solver(initial)
