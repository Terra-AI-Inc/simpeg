import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
from discretize import TensorMesh, TreeMesh
from discretize.utils import Zero

from ...base.pde_simulation import _inner_mat_mul_op
from ...utils import mkvc, validate_string
from ... import maps
from ..frequency_domain.simulation import BaseFDEMSimulation, Simulation3DElectricField
from ..frequency_domain.survey import Survey
from ..utils import omega
from .sources import Planewave
from .utils.source_utils import primary_e_1d_solution
from .receivers import Impedance, Tipper, Admittance
from .fields import (
    Fields1DPrimarySecondary,
    Fields1DElectricField,
    Fields1DMagneticField,
    Fields2DElectricField,
    Fields2DMagneticField,
)


def _centers_to_widths(centers):
    centers = np.asarray(centers)
    d = np.empty_like(centers)
    n = centers.shape[-1]
    d[..., 0] = 2 * centers[..., 0]
    for i in range(1, n):
        d[..., i] = 2 * (centers[..., i] - centers[..., i - 1]) - d[..., i - 1]
    return d


###################################
# 1D problems
###################################


def _bottom_robin_matrix(n, value):
    """Sparse (n, n) matrix with ``value`` in the entry for the bottom node."""
    return sp.csr_matrix(([value], ([0], [0])), shape=(n, n))


def _bottom_robin_deriv(prop_deriv, d_value, u, v, adjoint):
    """Derivative of the bottom Robin term times ``u`` with respect to the model.

    The term only depends on the property of the bottom cell (cell 0), through
    ``d_value``, and only acts on the bottom node (node 0).
    """
    # squeeze like _inner_mat_mul_op, so the shapes of the two parts match
    u = np.squeeze(np.asarray(u))
    v = np.squeeze(np.asarray(v))
    e0 = np.zeros(prop_deriv.shape[0])
    e0[0] = 1.0
    deriv_row = prop_deriv.T @ e0
    if adjoint:
        coef = d_value * u[0] * v[0]
        if u.ndim > 1:
            # sum over the fields stored in the columns of u
            coef = np.sum(coef)
        return np.multiply.outer(deriv_row, coef) if np.ndim(coef) else deriv_row * coef
    d_prop = deriv_row @ v
    out = np.zeros((u.shape[0],) + np.broadcast(u[0], d_prop).shape, dtype=complex)
    out[0] = d_value * u[0] * d_prop
    return out


class Simulation1DElectricField(BaseFDEMSimulation):
    r"""
    1D finite volume simulation for the natural source electromagnetic problem.

    This corresponds to the TE mode 2D simulation where the electric field is
    located at cell centers and the magnetic flux is on edges.

    We are solving the discrete version of

    .. math::

        \partial_z E_y = i \omega \mu_0 H_x = 0

        sigma E_y = \partial_z H_x

    with default boundary conditions that $H_x[z_max] = 1$ (a plane wave source at
    the top of the domain), and $H_x[z_min] = 0$.

    When we discretize, we obtain:

    where the Magnetic field is defined on edges, and the electric field is
    defined on cell centers.
    """

    _solutionType = "eSolution"
    _formulation = "EB"  # electric-field component is on cell-centers
    fieldsPair = Fields1DElectricField

    def __init__(self, mesh, **kwargs):
        if mesh.dim > 1:
            raise ValueError(
                f"The mesh must be a 1D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)

        self._rhs = mesh.boundary_node_vector_integral * [0 + 0j, 1 + 0j]

    def getA(self, freq):
        r"""
        System matrix

        .. math::

            \mathbf{A} =
                \mathbf{G}^\top \mathbf{M}^e_{\mu^{-1}} \mathbf{G}
                + 1\omega \mathbf{M}^f_\sigma
        """

        G = self.mesh.nodal_gradient
        MeMui = self.MeMui
        MfSigma = self.MfSigma

        A = G.T.tocsr() @ MeMui @ G + 1j * omega(freq) * MfSigma
        return A + _bottom_robin_matrix(A.shape[0], self._bottom_robin(freq)[0])

    def _bottom_robin(self, freq):
        r"""Robin (downgoing plane wave) term at the bottom node.

        :math:`i k / \mu` with :math:`k = \sqrt{-i \omega \mu \sigma}` for the
        bottom cell, and its derivatives with respect to the bottom cell's
        conductivity and inverse permeability.
        """
        sigma = np.atleast_1d(self.sigma)[0]
        mui = np.atleast_1d(self.mui)[0]
        value = 1j * np.sqrt(-1j * omega(freq) * sigma * mui)
        return value, value / (2 * sigma), value / (2 * mui)

    def getADeriv_sigma(self, freq, u, v, adjoint=False):
        dA_v = 1j * omega(freq) * self.MfSigmaDeriv(u, v, adjoint=adjoint)
        if self.sigmaMap is None:
            return dA_v
        d_value = self._bottom_robin(freq)[1]
        return dA_v + _bottom_robin_deriv(self.sigmaDeriv, d_value, u, v, adjoint)

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        G = self.mesh.nodal_gradient
        if adjoint:
            dA_v = self.MeMuiDeriv(G * u, G * v, adjoint)
        else:
            dA_v = G.T * self.MeMuiDeriv(G * u, v, adjoint)
        if self.muiMap is None:
            return dA_v
        d_value = self._bottom_robin(freq)[2]
        return dA_v + _bottom_robin_deriv(self.muiDeriv, d_value, u, v, adjoint)

    def getRHS(self, freq):
        """
        Right hand side constructed using Dirichlet boundary conditions
        """
        return 1j * omega(freq) * self._rhs

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        return Zero()

    def getADeriv(self, freq, u, v, adjoint=False):
        return self.getADeriv_sigma(freq, u, v, adjoint) + self.getADeriv_mui(
            freq, u, v, adjoint
        )

    def getJ(self, m, f=None):
        r"""Generate the full sensitivity matrix.

        .. important::

            This method hasn't been implemented yet for this class.

        Raises
        ------
        NotImplementedError
        """
        msg = (
            "The getJ method hasn't been implemented for the "
            f"{type(self).__name__} yet."
        )
        raise NotImplementedError(msg)


class Simulation1DMagneticField(BaseFDEMSimulation):
    """
    1D finite volume simulation for the natural source electromagnetic problem.

    This corresponds to the TM mode 2D simulation where the magnetic field is
    located at faces (nodes) and the electric field is on edges (cell_centers).
    """

    _solutionType = "hSolution"
    _formulation = "HJ"
    fieldsPair = Fields1DMagneticField

    def __init__(self, mesh, **kwargs):
        if mesh.dim > 1:
            raise ValueError(
                f"The mesh must be a 1D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)

        # corresponds to a dirichlet boundaries at the top (= 1 ) and bottom(=0)
        # for the y component of electric field
        self._rhs = -mesh.boundary_node_vector_integral * [0 + 0j, 1 + 0j]

    def getA(self, freq):
        """
        system matrix
        """
        G = self.mesh.nodal_gradient
        MeRho = self.MeRho
        MnMu = self.MnMu

        A = G.T.tocsr() @ MeRho @ G + 1j * omega(freq) * MnMu
        return A + _bottom_robin_matrix(A.shape[0], self._bottom_robin(freq)[0])

    def _bottom_robin(self, freq):
        r"""Robin (downgoing plane wave) term at the bottom node.

        :math:`i k \rho` with :math:`k = \sqrt{-i \omega \mu \sigma}` for the
        bottom cell, and its derivatives with respect to the bottom cell's
        resistivity and permeability.
        """
        rho = np.atleast_1d(self.rho)[0]
        mu = np.atleast_1d(self.mu)[0]
        value = 1j * np.sqrt(-1j * omega(freq) * mu * rho)
        return value, value / (2 * rho), value / (2 * mu)

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        G = self.mesh.nodal_gradient
        if adjoint:
            dA_v = self.MeRhoDeriv(G * u, G * v, adjoint)
        else:
            dA_v = G.T * self.MeRhoDeriv(G * u, v, adjoint)
        if self.rhoMap is None:
            return dA_v
        d_value = self._bottom_robin(freq)[1]
        return dA_v + _bottom_robin_deriv(self.rhoDeriv, d_value, u, v, adjoint)

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        MnMuDeriv = self.MnMuDeriv(u)
        if adjoint is True:
            dA_v = 1j * omega(freq) * (MnMuDeriv.T * v)
        else:
            dA_v = 1j * omega(freq) * (MnMuDeriv * v)
        if self.muMap is None:
            return dA_v
        d_value = self._bottom_robin(freq)[2]
        return dA_v + _bottom_robin_deriv(self.muDeriv, d_value, u, v, adjoint)

    def getRHS(self, freq):
        """
        right hand side
        """
        return self._rhs

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        return Zero()

    def getADeriv(self, freq, u, v, adjoint=False):
        return self.getADeriv_rho(freq, u, v, adjoint) + self.getADeriv_mu(
            freq, u, v, adjoint
        )

    def getJ(self, m, f=None):
        r"""Generate the full sensitivity matrix.

        .. important::

            This method hasn't been implemented yet for this class.

        Raises
        ------
        NotImplementedError
        """
        msg = (
            "The getJ method hasn't been implemented for the "
            f"{type(self).__name__} yet."
        )
        raise NotImplementedError(msg)


class Simulation1DPrimarySecondary(Simulation1DElectricField):
    r"""
    A NSEM problem solving a e formulation and primary/secondary fields decomposition.

    By eliminating the magnetic flux density using

    .. math ::

        \mathbf{b} = \frac{1}{i \omega} \left(-\mathbf{C} \mathbf{e} \right)


    we can write Maxwell's equations as a second order system in
    :math:`\mathbf{e}` only:

    .. math ::

        \left[
            \mathbf{C}^{\top} \mathbf{M_{\mu^{-1}}^e } \mathbf{C}
            + i \omega \mathbf{M_{\sigma}^f}
        \right]
        \mathbf{e}_{s}
        = i \omega \mathbf{M_{\sigma_{s}}^f } \mathbf{e}_{p}

    which we solve for :math:`\mathbf{e_s}`.
    The total field :math:`\mathbf{e} = \mathbf{e_p} + \mathbf{e_s}`.

    The primary field is estimated from a background model (commonly half space ).
    """

    fieldsPair = Fields1DPrimarySecondary

    def __init__(self, mesh, survey=None, sigmaPrimary=None, **kwargs):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.sigmaPrimary = sigmaPrimary

    @property
    def sigmaPrimary(self):
        """
        A background model, use for the calculation of the primary fields.

        """
        return self._sigmaPrimary

    @sigmaPrimary.setter
    def sigmaPrimary(self, val):
        # Note: TODO add logic for val, make sure it is the correct size.
        self._sigmaPrimary = val

    def getADeriv(self, freq, u, v, adjoint=False):
        """
        The derivative of A wrt sigma
        """
        # Only select the yx polarization
        return super().getADeriv(freq, u[:, 1], v, adjoint=adjoint)

    def getRHS(self, freq):
        """
        Function to return the right hand side for the system.

        :param float freq: Frequency
        :rtype: numpy.ndarray
        :return: RHS for 1 polarizations, primary fields (nF, 1)
        """

        # Get sources for the frequncy(polarizations)
        src = self.survey.get_sources_by_frequency(freq)[0]
        # Only select the yx polarization
        S_e = mkvc(src.s_e(self)[:, 1], 2)
        return -1j * omega(freq) * S_e

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        """
        The derivative of the RHS wrt sigma
        """
        S_eDeriv = src.s_eDeriv_m(self, v, adjoint)
        return -1j * omega(freq) * S_eDeriv


###################################
# 2D problems
###################################
def _bottom_robin_ops_2d(mesh):
    """Operators for a Robin term on the bottom boundary edges of a 2D mesh.

    Returns
    -------
    P : (n_bottom_edges, n_edges) scipy.sparse.csr_matrix
        Selects the edges on the bottom boundary.
    K : (n_bottom_edges, n_cells) scipy.sparse.csr_matrix
        The length of each bottom edge, at the cell above it.
    """
    edges = np.r_[mesh.edges_x, mesh.edges_y]
    bottom = np.where(
        np.isclose(edges[:, 1], mesh.nodes_y[0])
        & (np.arange(len(edges)) < mesh.n_edges_x)
    )[0]
    n_bot = len(bottom)
    P = sp.csr_matrix(
        (np.ones(n_bot), (np.arange(n_bot), bottom)), shape=(n_bot, mesh.n_edges)
    )
    eps = 1e-3 * mesh.h[1].min()
    cells = mesh.point2index(edges[bottom] + np.r_[0.0, eps])
    K = sp.csr_matrix(
        (mesh.edge_lengths[bottom], (np.arange(n_bot), cells)),
        shape=(n_bot, mesh.n_cells),
    )
    return P, K


def _bottom_robin_deriv_2d(ops, scale, dg_dm, u, v, adjoint):
    """Model derivative of ``scale * P.T @ diag(K @ g) @ P @ u``."""
    P, K = ops
    dKg_dm = K @ dg_dm
    if adjoint:
        return _inner_mat_mul_op(dKg_dm, P @ u, P @ (scale * v), adjoint=True)
    return scale * (P.T @ _inner_mat_mul_op(dKg_dm, P @ u, v))


class Simulation2DElectricField(BaseFDEMSimulation):
    """
    A
    """

    _solutionType = "eSolution"
    _formulation = "EB"
    fieldsPair = Fields2DElectricField

    def __init__(self, mesh, h_bc=None, **kwargs):
        if mesh.dim != 2:
            raise ValueError(
                f"The mesh must be a 2D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)

        for src in self.survey.source_list:
            for rx in src.receiver_list:
                if not (
                    (rx.orientation == "xy" and isinstance(rx, Impedance))
                    or (rx.orientation == "yx" and isinstance(rx, Admittance))
                ):
                    raise TypeError(
                        "natural_source.Simulation2DElectricField only supports Impedance for"
                        " an xy receiver orientation OR Admittance for a yx receiver"
                        " orientation. Please provide a survey with valid receivers."
                    )

        if h_bc is None:
            if isinstance(mesh, (TensorMesh, TreeMesh)):
                b_e = mesh.boundary_edges
                top = np.where(b_e[:, 1] == mesh.nodes_y[-1])
                bot = np.where(b_e[:, 1] == mesh.nodes_y[0])
                left = np.where(b_e[:, 0] == mesh.nodes_x[0])
                right = np.where(b_e[:, 0] == mesh.nodes_x[-1])

                if isinstance(mesh, TensorMesh):
                    h_l = h_r = mesh.h[1]
                    is_b = np.zeros(mesh.shape_cells, dtype=bool)
                    is_b[0, :] = True
                    P_l = maps.Projection(mesh.n_cells, is_b.reshape(-1, order="F"))
                    is_b[0, :] = False
                    is_b[-1, :] = True
                    P_r = maps.Projection(mesh.n_cells, is_b.reshape(-1, order="F"))
                else:
                    h_l = _centers_to_widths(b_e[left][:, 1])
                    h_r = _centers_to_widths(b_e[right][:, 1])
                    b_l, b_r, _, __ = mesh.cell_boundary_indices
                    P_l = maps.Projection(mesh.n_cells, b_l)
                    P_r = maps.Projection(mesh.n_cells, b_r)

                self._b_inds = (left, right, bot, top)
                self._P_l = P_l
                self._P_r = P_r

                map_l_kwargs = {}
                map_r_kwargs = {}
                if self.sigmaMap is not None:
                    map_l_kwargs["sigmaMap"] = P_l * self.sigmaMap
                    map_r_kwargs["sigmaMap"] = P_r * self.sigmaMap
                if self.muiMap is not None:
                    map_l_kwargs["muiMap"] = P_l * self.muiMap
                    map_r_kwargs["muiMap"] = P_r * self.muiMap

                # create a survey with 1 source per frequency (no receivers)
                frequencies = self.survey.frequencies
                survey = Survey([Planewave([], freq) for freq in frequencies])
                self._sim_left = Simulation1DElectricField(
                    TensorMesh((h_l,), (mesh.nodes_y[0],)),
                    survey=survey,
                    solver=self.solver,
                    **map_l_kwargs,
                )
                self._sim_right = Simulation1DElectricField(
                    TensorMesh((h_r,), (mesh.nodes_y[0],)),
                    survey=survey,
                    solver=self.solver,
                    **map_r_kwargs,
                )
            else:
                raise NotImplementedError(
                    f"Unable to infer 1D mesh from {type(mesh)}. You must supply custom"
                    " boundary conditions for the electric field."
                )
            self._h_bc = None
        else:
            n_be = mesh.boundary_edges.shape[0]
            for freq in self.survey.frequencies:
                try:
                    h = h_bc[freq]
                    if len(h) != n_be:
                        raise ValueError(
                            f"Boundary condition item for frequency {freq} is incorrect length."
                            f" Should be the same length as number of boundary_edges, {n_be}, "
                            f" saw a length of {len(h)}"
                        )
                except TypeError:
                    raise TypeError(
                        "h_bc must be a dictionary of numpy arrays indexed by frequency."
                    )
                except IndexError:
                    raise TypeError(
                        "h_bc must be a dictionary of numpy arrays indexed by frequency. Did not"
                        f" find key {freq}."
                    )
                except KeyError:
                    raise KeyError(
                        "h_bc must be a dictionary of numpy arrays indexed by frequency. Did not"
                        f" find key {freq}."
                    )
            self._h_bc = h_bc
        self._M_bc = mesh.boundary_edge_vector_integral

    def getA(self, freq):
        r"""
        System matrix

        .. math::

            \mathbf{A} =
                \mathbf{C}^\top \mathbf{M}^{cc}_{\mu} \mathbf{C}
                + 1\omega \mathbf{M}^e_\sigma

        """
        C = self.mesh.edge_curl
        Mcc_mui = self.MccMui
        Me_sigma = self.MeSigma

        A = C.T.tocsr() @ Mcc_mui @ C + 1j * omega(freq) * Me_sigma
        if self._h_bc is None:
            A = A + self._bottom_robin_matrix(freq)
        return A

    def _bottom_robin_values(self):
        sigma = np.broadcast_to(self.sigma, (self.mesh.n_cells,))
        mui = np.broadcast_to(self.mui, (self.mesh.n_cells,))
        return sigma, mui, np.sqrt(sigma * mui)

    def _bottom_robin_matrix(self, freq):
        r"""Robin (downgoing plane wave) term on the bottom boundary edges.

        :math:`\sqrt{i \omega \sigma / \mu}` of the bottom cells, integrated
        along the bottom boundary.
        """
        P, K = self._bottom_robin_ops
        scale = np.sqrt(1j * omega(freq))
        return scale * (P.T @ sp.diags(K @ self._bottom_robin_values()[2]) @ P)

    @property
    def _bottom_robin_ops(self):
        if getattr(self, "_bottom_robin_ops_cache", None) is None:
            self._bottom_robin_ops_cache = _bottom_robin_ops_2d(self.mesh)
        return self._bottom_robin_ops_cache

    def getRHS(self, freq):
        """
        Right hand side constructed using Dirichlet boundary conditions
        """
        M_bc = self._M_bc
        if self._h_bc is None:
            # left and right have the same 1D survey
            src = self._sim_left.survey.get_sources_by_frequency(freq)[0]
            f_left, f_right = self.boundary_fields()
            h_bc = np.zeros(M_bc.shape[1], dtype=complex)
            left, right, bot, top = self._b_inds
            h_bc[top] = 1.0
            h_bc[left] = f_left[src, "h"][:, 0]
            h_bc[right] = f_right[src, "h"][:, 0]
        else:
            h_bc = self._h_bc[freq]
        return 1j * omega(freq) * (M_bc @ h_bc)

    def getADeriv_sigma(self, freq, u, v, adjoint=False):
        dA_v = 1j * omega(freq) * self.MeSigmaDeriv(u, v, adjoint=adjoint)
        if self._h_bc is not None or self.sigmaMap is None:
            return dA_v
        sigma, _, g = self._bottom_robin_values()
        dg_dm = sp.diags(0.5 * g / sigma) @ self.sigmaDeriv
        scale = np.sqrt(1j * omega(freq))
        return dA_v + _bottom_robin_deriv_2d(
            self._bottom_robin_ops, scale, dg_dm, u, v, adjoint
        )

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        C = self.mesh.edge_curl
        if adjoint:
            dA_v = self.MccMuiDeriv(C * u, C * v, adjoint)
        else:
            dA_v = C.T * self.MccMuiDeriv(C * u, v, adjoint)
        if self._h_bc is not None or self.muiMap is None:
            return dA_v
        _, mui, g = self._bottom_robin_values()
        dg_dm = sp.diags(0.5 * g / mui) @ self.muiDeriv
        scale = np.sqrt(1j * omega(freq))
        return dA_v + _bottom_robin_deriv_2d(
            self._bottom_robin_ops, scale, dg_dm, u, v, adjoint
        )

    def getADeriv(self, freq, u, v, adjoint=False):
        return self.getADeriv_sigma(freq, u, v, adjoint) + self.getADeriv_mui(
            freq, u, v, adjoint
        )

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        if self._h_bc is not None:
            return Zero()
        M_bc = self._M_bc
        f_left, f_right = self.boundary_fields()
        left, right, _, __ = self._b_inds
        src_1d = self._sim_left.survey.get_sources_by_frequency(freq)[0]

        # derivatives from the Jv func of the 1D sim
        if not adjoint:
            h_bc_dm_v = np.zeros(M_bc.shape[1], dtype=complex)
            h_bc_dm_v[left] = f_left.field_deriv_m("h", freq, src_1d, v, adjoint=False)
            h_bc_dm_v[right] = f_right.field_deriv_m(
                "h", freq, src_1d, v, adjoint=False
            )

            return 1j * omega(freq) * (M_bc @ h_bc_dm_v)
        else:
            v_dm = M_bc.T @ v
            v_left, v_right = v_dm[left], v_dm[right]
            df_dmT = f_left.field_deriv_m("h", freq, src_1d, v_left, adjoint=True)
            df_dmT += f_right.field_deriv_m("h", freq, src_1d, v_right, adjoint=True)

            return 1j * omega(freq) * df_dmT

    def boundary_fields(self, model=None):
        "Returns the 1D field objects at the boundaries"
        if getattr(self, "_boundary_fields", None) is None:
            if model is None:
                model = self.model
            sim = self._sim_left
            if self.muiMap is None:
                try:
                    sim.mui = self._P_l @ self.mui
                except Exception:
                    sim.mui = self.mui
            if self.sigmaMap is None:
                try:
                    sim.sigma = self._P_l @ self.sigma
                except Exception:
                    sim.sigma = self.sigma
            f_left = sim.fields(model)

            sim = self._sim_right
            if self.muiMap is None:
                try:
                    sim.mui = self._P_r @ self.mui
                except Exception:
                    sim.mui = self.mui
            if self.sigmaMap is None:
                try:
                    sim.sigma = self._P_r @ self.sigma
                except Exception:
                    sim.sigma = self.sigma
            f_right = sim.fields(model)

            self._boundary_fields = (f_left, f_right)
        return self._boundary_fields

    @property
    def _delete_on_model_update(self):
        items = super()._delete_on_model_update
        items.append("_boundary_fields")
        return items


class Simulation2DMagneticField(BaseFDEMSimulation):
    """
    A
    """

    _solutionType = "hSolution"
    _formulation = "HJ"
    fieldsPair = Fields2DMagneticField

    def __init__(self, mesh, e_bc=None, **kwargs):
        if mesh.dim != 2:
            raise ValueError(
                f"The mesh must be a 2D mesh. The provided mesh has dimension {mesh.dim}"
            )

        super().__init__(mesh, **kwargs)

        for src in self.survey.source_list:
            for rx in src.receiver_list:
                if not (
                    (rx.orientation == "yx" and isinstance(rx, Impedance))
                    or (rx.orientation == "xy" and isinstance(rx, Admittance))
                    or (rx.orientation == "zx" and isinstance(rx, Tipper))
                ):
                    raise TypeError(
                        "natural_source.Simulation2DMagneticField supports Impedance and"
                        " Admittance receivers for an yx orientation, and Tipper receivers"
                        " for a zx orientation. Please provide a survey with valid receivers."
                    )

        if e_bc is None:
            if isinstance(mesh, (TensorMesh, TreeMesh)):
                b_e = mesh.boundary_edges
                top = np.where(b_e[:, 1] == mesh.nodes_y[-1])
                bot = np.where(b_e[:, 1] == mesh.nodes_y[0])
                left = np.where(b_e[:, 0] == mesh.nodes_x[0])
                right = np.where(b_e[:, 0] == mesh.nodes_x[-1])

                if isinstance(mesh, TensorMesh):
                    h_l = h_r = mesh.h[1]
                    is_b = np.zeros(mesh.shape_cells, dtype=bool)
                    is_b[0, :] = True
                    P_l = maps.Projection(mesh.n_cells, is_b.reshape(-1, order="F"))
                    is_b[0, :] = False
                    is_b[-1, :] = True
                    P_r = maps.Projection(mesh.n_cells, is_b.reshape(-1, order="F"))
                else:
                    h_l = _centers_to_widths(b_e[left][:, 1])
                    h_r = _centers_to_widths(b_e[right][:, 1])
                    b_l, b_r, _, __ = mesh.cell_boundary_indices
                    P_l = maps.Projection(mesh.n_cells, b_l)
                    P_r = maps.Projection(mesh.n_cells, b_r)

                self._b_inds = (left, right, bot, top)
                self._P_l = P_l
                self._P_r = P_r

                map_l_kwargs = {}
                map_r_kwargs = {}
                if self.rhoMap is not None:
                    map_l_kwargs["rhoMap"] = P_l * self.rhoMap
                    map_r_kwargs["rhoMap"] = P_r * self.rhoMap
                if self.muMap is not None:
                    map_l_kwargs["muMap"] = P_l * self.muMap
                    map_r_kwargs["muMap"] = P_r * self.muMap

                # create a survey with 1 source per frequency (no receivers)
                frequencies = self.survey.frequencies
                survey = Survey([Planewave([], freq) for freq in frequencies])
                self._sim_left = Simulation1DMagneticField(
                    TensorMesh((h_l,), (mesh.nodes_y[0],)),
                    survey=survey,
                    solver=self.solver,
                    **map_l_kwargs,
                )
                self._sim_right = Simulation1DMagneticField(
                    TensorMesh((h_r,), (mesh.nodes_y[0],)),
                    survey=survey,
                    solver=self.solver,
                    **map_r_kwargs,
                )
            else:
                raise NotImplementedError(
                    f"Unable to infer 1D mesh from {type(mesh)}. You must supply custom"
                    " boundary conditions for the electric field."
                )
            self._e_bc = None
        else:
            n_be = mesh.boundary_edges.shape[0]
            for freq in self.survey.frequencies:
                try:
                    e = e_bc[freq]
                    if len(e) != n_be:
                        raise ValueError(
                            f"Boundary condition item for frequency {freq} is incorrect length."
                            f" Should be the same length as number of boundary_edges, {n_be}, "
                            f" saw a length of {len(e)}"
                        )
                except TypeError:
                    raise TypeError(
                        "e_bc must be a dictionary of numpy arrays indexed by frequency."
                    )
                except IndexError:
                    raise TypeError(
                        "e_bc must be a dictionary of numpy arrays indexed by frequency."
                    )
                except KeyError:
                    raise KeyError(
                        "e_bc must be a dictionary of numpy arrays indexed by frequency. Did not"
                        f" find key {freq}."
                    )
            self._e_bc = e_bc
        self._M_bc = mesh.boundary_edge_vector_integral

    def getA(self, freq):
        r"""
        System matrix

        .. math::

            \mathbf{A} =
                \mathbf{C}^\top \mathbf{M}^{cc}_{\rho} \mathbf{C}
                + 1\omega \mathbf{M}^e_\mu
        """
        C = self.mesh.edge_curl
        Mcc_rho = self.MccRho
        Me_mu = self.MeMu

        A = C.T.tocsr() @ Mcc_rho @ C + 1j * omega(freq) * Me_mu
        if self._e_bc is None:
            A = A + self._bottom_robin_matrix(freq)
        return A

    def _bottom_robin_values(self):
        rho = np.broadcast_to(self.rho, (self.mesh.n_cells,))
        mu = np.broadcast_to(self.mu, (self.mesh.n_cells,))
        return rho, mu, np.sqrt(rho * mu)

    def _bottom_robin_matrix(self, freq):
        r"""Robin (downgoing plane wave) term on the bottom boundary edges.

        :math:`i \sqrt{-i \omega \mu \rho}` of the bottom cells, integrated
        along the bottom boundary.
        """
        P, K = self._bottom_robin_ops
        scale = 1j * np.sqrt(-1j * omega(freq))
        return scale * (P.T @ sp.diags(K @ self._bottom_robin_values()[2]) @ P)

    @property
    def _bottom_robin_ops(self):
        if getattr(self, "_bottom_robin_ops_cache", None) is None:
            self._bottom_robin_ops_cache = _bottom_robin_ops_2d(self.mesh)
        return self._bottom_robin_ops_cache

    def getRHS(self, freq):
        """
        Right hand side constructed using Dirichlet boundary conditions
        """
        M_bc = self._M_bc
        if self._e_bc is None:
            # left and right have the same 1D survey
            src = self._sim_left.survey.get_sources_by_frequency(freq)[0]
            f_left, f_right = self.boundary_fields()
            e_bc = np.zeros(M_bc.shape[1], dtype=complex)
            left, right, bot, top = self._b_inds
            e_bc[top] = 1.0
            e_bc[left] = f_left[src, "e"][:, 0]
            e_bc[right] = f_right[src, "e"][:, 0]
        else:
            e_bc = self._e_bc[freq]
        return -M_bc @ e_bc

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        C = self.mesh.edge_curl
        if adjoint:
            dA_v = self.MccRhoDeriv(C * u, C * v, adjoint)
        else:
            dA_v = C.T * self.MccRhoDeriv(C * u, v, adjoint)
        if self._e_bc is not None or self.rhoMap is None:
            return dA_v
        rho, _, g = self._bottom_robin_values()
        dg_dm = sp.diags(0.5 * g / rho) @ self.rhoDeriv
        scale = 1j * np.sqrt(-1j * omega(freq))
        return dA_v + _bottom_robin_deriv_2d(
            self._bottom_robin_ops, scale, dg_dm, u, v, adjoint
        )

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        dA_v = 1j * omega(freq) * self.MeMuDeriv(u, v, adjoint=adjoint)
        if self._e_bc is not None or self.muMap is None:
            return dA_v
        _, mu, g = self._bottom_robin_values()
        dg_dm = sp.diags(0.5 * g / mu) @ self.muDeriv
        scale = 1j * np.sqrt(-1j * omega(freq))
        return dA_v + _bottom_robin_deriv_2d(
            self._bottom_robin_ops, scale, dg_dm, u, v, adjoint
        )

    def getADeriv(self, freq, u, v, adjoint=False):
        return self.getADeriv_rho(freq, u, v, adjoint) + self.getADeriv_mu(
            freq, u, v, adjoint
        )

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        if self._e_bc is not None:
            return Zero()
        M_bc = self._M_bc
        f_left, f_right = self.boundary_fields()
        left, right, _, __ = self._b_inds
        src_1d = self._sim_left.survey.get_sources_by_frequency(freq)[0]

        # derivatives from the Jv func of the 1D sim
        if not adjoint:
            e_bc_dm_v = np.zeros(M_bc.shape[1], dtype=complex)
            e_bc_dm_v[left] = f_left.field_deriv_m("e", freq, src_1d, v, adjoint=False)
            e_bc_dm_v[right] = f_right.field_deriv_m(
                "e", freq, src_1d, v, adjoint=False
            )
            return -(M_bc @ e_bc_dm_v)
        else:
            v_dm = -(M_bc.T @ v)
            v_left, v_right = v_dm[left], v_dm[right]
            df_dmT = f_left.field_deriv_m("e", freq, src_1d, v_left, adjoint=True)
            df_dmT += f_right.field_deriv_m("e", freq, src_1d, v_right, adjoint=True)
            return df_dmT

    def boundary_fields(self, model=None):
        "Returns the 1D field objects at the boundaries"
        if getattr(self, "_boundary_fields", None) is None:
            if model is None:
                model = self.model
            sim = self._sim_left
            if self.muMap is None:
                try:
                    sim.mu = self._P_l @ self.mu
                except Exception:
                    sim.mu = self.mu
            if self.rhoMap is None:
                try:
                    sim.rho = self._P_l @ self.rho
                except Exception:
                    sim.rho = self.rho
            f_left = sim.fields(model)

            sim = self._sim_right
            if self.muMap is None:
                try:
                    sim.mu = self._P_r @ self.mu
                except Exception:
                    sim.mu = self.mu
            if self.rhoMap is None:
                try:
                    sim.rho = self._P_r @ self.rho
                except Exception:
                    sim.rho = self.rho
            f_right = sim.fields(model)

            self._boundary_fields = (f_left, f_right)
        return self._boundary_fields

    @property
    def _delete_on_model_update(self):
        items = super()._delete_on_model_update
        items.append("_boundary_fields")
        return items


###################################
# 3D problems
###################################


class Simulation3DPrimarySecondary(Simulation3DElectricField):
    r"""
    A NSEM problem solving a e formulation and a primary/secondary fields decomposition.

    By eliminating the magnetic flux density using

    .. math ::

        \mathbf{b} = \frac{1}{i \omega} \left(-\mathbf{C} \mathbf{e} \right)


    we can write Maxwell's equations as a second order system in
    :math:`\mathbf{e}` only:

    .. math ::

        \left[
            \mathbf{C}^{\top} \mathbf{M_{\mu^{-1}}^f} \mathbf{C}
            + i \omega \mathbf{M_{\sigma}^e}
        \right]
        \mathbf{e}_{s}
        = i \omega \mathbf{M_{\sigma_{p}}^e} \mathbf{e}_{p}

    which we solve for :math:`\mathbf{e_s}`.
    The total field :math:`\mathbf{e} = \mathbf{e_p} + \mathbf{e_s}`.

    The primary field is estimated from a background model (commonly as a 1D model).

    Parameters
    ----------
    mesh : discretize.TensorMesh or discretize.TreeMesh
        A 3D mesh.
    survey : .natural_source.Survey, optional
        The natural source survey.
    sigmaPrimary : float or numpy.ndarray, optional
        Background conductivity model used to compute the primary field.
    boundary_condition : {"robin", "robin_1d", "natural"}
        Boundary condition applied to the secondary field on the lateral
        (:math:`x` and :math:`y`) and bottom boundaries of the mesh. See the
        Notes.
    **kwargs
        Additional keyword arguments for
        :class:`~simpeg.electromagnetics.frequency_domain.Simulation3DElectricField`.

    Notes
    -----
    The weak form of the system drops the boundary integral

    .. math::

        -i \omega \oint_{\partial \Omega} \vec{w} \cdot
        (\hat{n} \times \vec{h}_s) \, dA,

    which imposes the natural condition
    :math:`\hat{n} \times \vec{h}_s = 0` on the secondary field. It is only
    accurate if the secondary field has decayed by the time it reaches the
    boundary, which requires the conductivity near the boundary to match the
    primary model and the mesh to be padded far enough away from any
    structure.

    With ``boundary_condition="robin"`` (the default), a first order
    absorbing (impedance) condition is used on the lateral and bottom
    boundaries instead,

    .. math::

        \hat{n} \times \vec{h}_s = - Y \vec{e}_{s, t},
        \quad
        Y = \frac{k}{\omega \mu_0} = \sqrt{\frac{\hat{\sigma}}{i \omega \mu_0}},

    where :math:`\vec{e}_{s, t}` is the tangential secondary electric field
    and :math:`\hat{\sigma}` is the conductivity (or admittivity) of the cell
    adjacent to each boundary face. This adds

    .. math::

        \sqrt{\frac{i \omega}{\mu_0}}
        \mathbf{P}^\top
        \textrm{diag} \left( \mathbf{K} \sqrt{\boldsymbol{\hat{\sigma}}} \right)
        \mathbf{P}

    to the system matrix, where :math:`\mathbf{P}` projects edges to the
    boundary edges and :math:`\mathbf{K}` integrates the cell values over
    each boundary edge's portion of the adjacent lateral and bottom boundary
    faces.
    This absorbs the outgoing secondary field instead of reflecting it, which
    reduces the amount of padding that is needed and relaxes the requirement
    that the conductivity model match the primary model at the lateral
    boundaries.

    Where the conductivity of a boundary column differs from the primary
    model, the secondary field there also contains the difference between the
    local 1D solution and the primary field, which travels vertically rather
    than out through the boundary. With ``boundary_condition="robin_1d"``,
    the absorbing condition is applied only to the part of the secondary
    field that remains after removing this difference field,
    :math:`\vec{e}_d = \vec{e}_{1D} - \vec{e}_p`,

    .. math::

        \hat{n} \times \vec{h}_s = - Y \left(\vec{e}_{s, t} - \vec{e}_{d, t}\right)
        + \hat{n} \times \vec{h}_d,

    where :math:`\vec{e}_{1D}` is the 1D solution for the conductivity along
    each lateral boundary face, scaled to have the same magnetic field as the
    primary field at the top of the mesh. The system matrix is the same as for
    ``"robin"``, and the known boundary data are added to the right hand side.
    This is exact when the conductivity is laterally uniform at the
    boundaries, even if it differs from the primary model. Structures that
    vary along a boundary face (for example topography or a contact that
    crosses the boundary) are only approximated by the local 1D fields.

    At the bottom boundary, the secondary field travels downward, which the
    Robin condition absorbs (this includes the 1D difference field, so the
    bottom boundary does not need the 1D boundary data). The top boundary
    always uses the natural condition, since in the air the conductivity is
    so small that the impedance term is negligible.
    """

    def __init__(
        self,
        mesh,
        survey=None,
        sigmaPrimary=None,
        boundary_condition="robin",
        **kwargs,
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.sigmaPrimary = sigmaPrimary
        self.boundary_condition = boundary_condition

    # fieldsPair = Fields3DPrimarySecondary

    @property
    def sigmaPrimary(self):
        """
        A background model, use for the calculation of the primary fields.

        """
        return self._sigmaPrimary

    @sigmaPrimary.setter
    def sigmaPrimary(self, val):
        # Note: TODO add logic for val, make sure it is the correct size.
        self._sigmaPrimary = val

    @property
    def boundary_condition(self):
        """Boundary condition on the lateral and bottom boundaries for the secondary field.

        Returns
        -------
        {"robin", "robin_1d", "natural"}
        """
        return self._boundary_condition

    @boundary_condition.setter
    def boundary_condition(self, value):
        self._boundary_condition = validate_string(
            "boundary_condition", value, ["robin", "robin_1d", "natural"]
        )
        self._robin_1d_cache = {}

    @property
    def _uses_robin(self):
        return self.boundary_condition in ("robin", "robin_1d")

    @property
    def _robin_boundary_operators(self):
        """Operators used to construct the Robin boundary term.

        Returns
        -------
        P : (n_boundary_edges, n_edges) scipy.sparse.csr_matrix
            Projection from edges to boundary edges.
        K : (n_boundary_edges, n_cells) scipy.sparse.csr_matrix
            Integrates a cell property over each boundary edge's share of the
            adjacent lateral and bottom boundary faces.
        """
        if getattr(self, "_robin_ops", None) is None:
            mesh = self.mesh
            P = mesh.project_edge_to_boundary_edge
            Pf = mesh.project_face_to_boundary_face
            # each boundary face contributes half of its area to each of its
            # four boundary edges.
            Av = 2 * Pf @ mesh.average_edge_to_face @ P.T
            # lateral and bottom faces (the top is left on the natural condition)
            is_robin = mesh.boundary_face_outward_normals[:, 2] < 0.5
            areas = (Pf @ mesh.face_areas) * is_robin
            # average_cell_to_face takes the adjacent cell value on boundary faces
            K = Av.T @ sp.diags(areas) @ Pf @ mesh.average_cell_to_face
            self._robin_ops = (P.tocsr(), K.tocsr())
        return self._robin_ops

    def _boundary_admittivity(self, freq):
        admittivity = self._get_admittivity(freq)
        if np.size(admittivity) == 1:
            admittivity = np.full(self.mesh.n_cells, admittivity)
        return admittivity

    def _robin_boundary_matrix(self, freq):
        """Robin boundary term added to the system matrix.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_edges, n_edges) scipy.sparse.csr_matrix
        """
        P, K = self._robin_boundary_operators
        admittivity = self._boundary_admittivity(freq)
        scale = np.sqrt(1j * omega(freq) / mu_0)
        return scale * (P.T @ sp.diags(K @ np.sqrt(admittivity)) @ P)

    def getA(self, freq):
        r"""System matrix for the frequency provided.

        .. math::
            \mathbf{A} = \mathbf{C^T M_{f\frac{1}{\mu}} C}
            + i\omega \mathbf{M_{e\sigma}} + \mathbf{B}

        where :math:`\mathbf{B}` is the Robin boundary term (zero when
        ``boundary_condition="natural"``). See the Notes section of
        :class:`Simulation3DPrimarySecondary`.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_edges, n_edges) sp.sparse.csr_matrix
            The system matrix.
        """
        A = super().getA(freq)
        if self._uses_robin:
            A = (A + self._robin_boundary_matrix(freq)).tocsr()
        return A

    def getADeriv_sigma(self, freq, u, v, adjoint=False):
        r"""Conductivity derivative operation for the system matrix times a vector.

        Includes the derivative of the Robin boundary term with respect to
        the conductivity of the cells on the lateral and bottom boundaries.

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        u : (n_edges,) numpy.ndarray
            The solution for the fields for the current model at the specified frequency.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges,) for the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            Derivative of system matrix times a vector. (n_edges,) for the standard operation.
            (n_param,) for the adjoint operation.
        """
        dA_v = super().getADeriv_sigma(freq, u, v, adjoint=adjoint)
        if not self._uses_robin or self.sigmaMap is None:
            return dA_v

        # B(sigma) @ u = scale * P.T @ diag(P @ u) @ K @ sqrt(sigma)
        P, K = self._robin_boundary_operators
        admittivity = self._boundary_admittivity(freq)
        scale = np.sqrt(1j * omega(freq) / mu_0)
        dsqrt_dm = K @ sp.diags(0.5 / np.sqrt(admittivity)) @ self.sigmaDeriv
        if adjoint:
            dB_v = _inner_mat_mul_op(dsqrt_dm, P @ u, P @ (scale * v), adjoint=True)
        else:
            dB_v = scale * (P.T @ _inner_mat_mul_op(dsqrt_dm, P @ u, v))
        return dA_v + dB_v

    @property
    def _clear_on_sigma_update(self):
        return super()._clear_on_sigma_update + ["_robin_1d_cache"]

    @property
    def _robin_1d_geometry(self):
        """Lateral boundary face and edge pairs and the 1D columns they sample.

        Each lateral boundary face samples the conductivity model along a
        vertical line through its center, just inside the boundary. Each
        (face, edge) pair represents one boundary edge's share of that face.
        """
        if getattr(self, "_robin_1d_geom", None) is None:
            mesh = self.mesh
            P = mesh.project_edge_to_boundary_edge.tocsr()
            Pf = mesh.project_face_to_boundary_face
            normals = mesh.boundary_face_outward_normals
            is_lateral = np.abs(normals[:, 2]) < 0.5

            pairs = (Pf @ mesh.average_edge_to_face @ P.T).tocoo()
            keep = is_lateral[pairs.row]
            face, edge = pairs.row[keep], P.indices[pairs.col[keep]]

            face_cell = np.asarray(
                (Pf @ mesh.average_cell_to_face).tocsr().argmax(axis=1)
            ).ravel()
            face_area = Pf @ mesh.face_areas
            face_xy = (Pf @ mesh.faces)[:, :2]

            # a vertical line through each lateral face, just inside the mesh
            eps = 1e-3 * min(h.min() for h in mesh.h[:2])
            lateral_faces = np.where(is_lateral)[0]
            line_xy = face_xy[lateral_faces] - eps * normals[lateral_faces, :2]
            line_xy, face_col = np.unique(line_xy, axis=0, return_inverse=True)
            face_column = np.full(len(normals), -1)
            face_column[lateral_faces] = face_col.ravel()

            z_cc = mesh.cell_centers_z
            points = np.column_stack(
                [
                    np.repeat(line_xy, len(z_cc), axis=0),
                    np.tile(z_cc, len(line_xy)),
                ]
            )
            column_cells = mesh.point2index(points).reshape(len(line_xy), len(z_cc))

            # position of each edge on the vertical nodes of the 1D columns
            z_nodes = mesh.nodes_z
            n_ex, n_ey = mesh.n_edges_x, mesh.n_edges_y
            edge_type = np.where(edge < n_ex, 0, np.where(edge < n_ex + n_ey, 1, 2))
            edge_z = np.r_[mesh.edges_x[:, 2], mesh.edges_y[:, 2], mesh.edges_z[:, 2]]
            half_length = np.where(edge_type == 2, mesh.edge_lengths[edge] / 2, 0.0)
            tol = 1e-6 * mesh.h[2].min()
            k_bot = np.searchsorted(z_nodes, edge_z[edge] - half_length - tol)
            k_top = np.searchsorted(z_nodes, edge_z[edge] + half_length - tol)

            self._robin_1d_geom = {
                "edge": edge,
                "cell": face_cell[face],
                "column": face_column[face],
                "normal": normals[face],
                "weight": 0.5 * face_area[face],
                "edge_type": edge_type,
                "k_bot": k_bot,
                "k_top": k_top,
                "column_cells": column_cells,
            }
        return self._robin_1d_geom

    def _robin_1d_columns(self, freq, src):
        """1D simulations and difference fields for the lateral boundary columns.

        Each unique column of conductivity along the lateral boundaries is solved
        with :class:`Simulation1DElectricField` on the same padded vertical
        discretization used for the primary field, so that a column matching the
        primary model gives exactly zero difference field.

        Returns
        -------
        dict
            ``"e_d"`` : (n_columns, n_z + 1) numpy.ndarray
                Difference field :math:`e_{1D} - e_p` on the vertical nodes of
                each boundary column.
            ``"groups"`` : list of dict
                One per unique column, with the 1D simulation and what is
                needed for derivatives.
        """
        key = (freq, id(src))
        cache = getattr(self, "_robin_1d_cache", None)
        if cache is None:
            cache = self._robin_1d_cache = {}
        if key in cache:
            return cache[key]

        mesh = self.mesh
        geom = self._robin_1d_geometry
        w = omega(freq)
        hz = mesh.h[2]
        n_z = len(hz)

        sigma = self.sigma
        if np.size(sigma) == 1:
            sigma = np.full(mesh.n_cells, sigma)
        column_sigma = sigma[geom["column_cells"]]

        sigma_1d, _ = src._get_sigmas(self)
        e_p = primary_e_1d_solution(mesh, sigma_1d, freq)

        # h in the top cell for an x-polarized field, as a linear function of e
        h_top = np.zeros(n_z + 1, dtype=complex)
        h_top[-1] = -1 / (hz[-1] * 1j * w * mu_0)
        h_top[-2] = 1 / (hz[-1] * 1j * w * mu_0)
        h_top_p = h_top @ e_p

        unique_sigma, inverse = np.unique(column_sigma, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        e_d = np.empty((len(geom["column_cells"]), n_z + 1), dtype=complex)
        groups = []
        survey_1d = Survey([Planewave([], freq)])
        for i, sig in enumerate(unique_sigma):
            # pad below by a few skin depths, as the primary field solver does
            skin_depth = np.sqrt(2 / (w * mu_0 * sig[0]))
            n_pad = int(np.ceil(3.0 * skin_depth / hz[0]))
            mesh_1d = TensorMesh(
                [np.pad(hz, (n_pad, 0), mode="edge")],
                origin=[mesh.origin[2] - hz[0] * n_pad],
            )
            pad_map = maps.Projection(
                n_z, np.r_[np.zeros(n_pad, dtype=int), np.arange(n_z)]
            )
            sim_1d = Simulation1DElectricField(
                mesh_1d, survey=survey_1d, sigmaMap=pad_map, solver=self.solver
            )
            fields_1d = sim_1d.fields(sig)
            u_1d = fields_1d[survey_1d.source_list[0], "e"][:, 0]
            e_1d = u_1d[n_pad:]
            # the difference field has the same h as the primary at the top
            scale = h_top_p / (h_top @ e_1d)
            columns = np.where(inverse == i)[0]
            e_d[columns] = scale * e_1d - e_p
            groups.append(
                {
                    "sim": sim_1d,
                    "u": u_1d,
                    "n_pad": n_pad,
                    "e": e_1d,
                    "scale": scale,
                    "columns": columns,
                }
            )

        cache[key] = {"e_d": e_d, "groups": groups, "h_top": h_top}
        return cache[key]

    def _robin_1d_pair_fields(self, freq, e_columns):
        """Boundary values of a set of column fields at each (face, edge) pair.

        Returns ``E`` along the horizontal edges, and the y component of h for
        the x polarization (``Hy``) on the vertical edges; the y polarization
        has h = -Hy x_hat.
        """
        geom = self._robin_1d_geometry
        z_nodes = self.mesh.nodes_z
        col, k_bot, k_top = geom["column"], geom["k_bot"], geom["k_top"]
        dz = np.where(geom["edge_type"] == 2, z_nodes[k_top] - z_nodes[k_bot], 1.0)
        E = e_columns[col, k_bot]
        Hy = -(e_columns[col, k_top] - E) / dz / (1j * omega(freq) * mu_0)
        return E, Hy, dz

    def _robin_1d_assemble(self, values):
        """Sum the values of each (face, edge) pair onto the edges."""
        edge = self._robin_1d_geometry["edge"]
        n = self.mesh.n_edges
        return np.bincount(edge, values.real, minlength=n) + 1j * np.bincount(
            edge, values.imag, minlength=n
        )

    def _robin_1d_boundary_data(self, freq, src):
        """Right hand side boundary data from the local 1D boundary fields.

        Returns
        -------
        (n_edges, 2) numpy.ndarray
            Boundary data for the x and y polarizations.
        """
        geom = self._robin_1d_geometry
        w = omega(freq)
        E, Hy, _ = self._robin_1d_pair_fields(
            freq, self._robin_1d_columns(freq, src)["e_d"]
        )
        admittivity = self._boundary_admittivity(freq)[geom["cell"]]
        a = np.sqrt(1j * w * admittivity / mu_0)
        is_vertical = geom["edge_type"] == 2
        rhs = np.zeros((self.mesh.n_edges, 2), dtype=complex)
        for pol in range(2):
            is_tangent = geom["edge_type"] == pol
            values = geom["weight"] * (
                np.where(is_tangent, a * E, 0.0)
                + np.where(is_vertical, 1j * w * geom["normal"][:, pol] * Hy, 0.0)
            )
            rhs[:, pol] = self._robin_1d_assemble(values)
        return rhs

    def _robin_1d_boundary_data_deriv(self, freq, src, v, adjoint=False):
        """Derivative of the boundary data with respect to the conductivity of the cells.

        Parameters
        ----------
        v : numpy.ndarray
            (n_cells,) for the standard operation, (n_edges, 2) or
            (n_edges, n, 2) for the adjoint operation.

        Returns
        -------
        numpy.ndarray
            (n_edges, 2) for the standard operation, (n_cells,) or
            (n_cells, n) for the adjoint operation.
        """
        mesh = self.mesh
        geom = self._robin_1d_geometry
        w = omega(freq)
        columns = self._robin_1d_columns(freq, src)
        h_top = columns["h_top"]
        cells = geom["column_cells"]
        n_z = cells.shape[1]

        E, _, dz = self._robin_1d_pair_fields(freq, columns["e_d"])
        admittivity = self._boundary_admittivity(freq)[geom["cell"]]
        a = np.sqrt(1j * w * admittivity / mu_0)
        da = 0.5 * a / admittivity
        weight, normal = geom["weight"], geom["normal"]
        is_vertical = geom["edge_type"] == 2
        col, k_bot, k_top = geom["column"], geom["k_bot"], geom["k_top"]

        if not adjoint:
            # perturbation of each column's difference field, through its 1D simulation
            de_d = np.zeros((cells.shape[0], n_z + 1), dtype=complex)
            for group in columns["groups"]:
                sim, n_pad = group["sim"], group["n_pad"]
                dsigma = v[cells[group["columns"]]].T  # (n_z, n_columns)
                du = -(sim.Ainv[0] * sim.getADeriv(freq, group["u"], dsigma))
                de = du.reshape(len(group["u"]), -1)[n_pad:]
                e = group["e"]
                de_d[group["columns"]] = (
                    group["scale"] * (de - np.outer(e, h_top @ de) / (h_top @ e))
                ).T

            dE, dHy, _ = self._robin_1d_pair_fields(freq, de_d)
            dsigma_cell = v[geom["cell"]]
            out = np.zeros((mesh.n_edges, 2), dtype=complex)
            for pol in range(2):
                is_tangent = geom["edge_type"] == pol
                values = weight * (
                    np.where(is_tangent, a * dE + da * dsigma_cell * E, 0.0)
                    + np.where(is_vertical, 1j * w * normal[:, pol] * dHy, 0.0)
                )
                out[:, pol] = self._robin_1d_assemble(values)
            return out

        v = np.asarray(v).reshape(mesh.n_edges, -1, 2)
        n_rhs = v.shape[1]
        dsigma = np.zeros((mesh.n_cells, n_rhs), dtype=complex)
        # sensitivity of v . rhs to each column's difference field
        g = np.zeros((cells.shape[0], n_z + 1, n_rhs), dtype=complex)
        c_h = 1j * w * weight / (dz * 1j * w * mu_0)
        for pol in range(2):
            vp = v[geom["edge"], :, pol]  # (n_pairs, n_rhs)
            is_tangent = (geom["edge_type"] == pol)[:, None]
            np.add.at(
                dsigma,
                geom["cell"],
                np.where(is_tangent, (weight * da * E)[:, None] * vp, 0.0),
            )
            np.add.at(
                g, (col, k_bot), np.where(is_tangent, (weight * a)[:, None] * vp, 0.0)
            )
            gh = np.where(
                is_vertical[:, None], (c_h * normal[:, pol])[:, None] * vp, 0.0
            )
            np.add.at(g, (col, k_top), -gh)
            np.add.at(g, (col, k_bot), gh)

        for group in columns["groups"]:
            sim, n_pad, e = group["sim"], group["n_pad"], group["e"]
            cols = group["columns"]
            g_e = (
                g[cols].transpose(1, 0, 2).reshape(n_z + 1, -1)
            )  # (n_z + 1, n_cols * n_rhs)
            g_e = group["scale"] * (g_e - np.outer(h_top, e @ g_e) / (h_top @ e))
            g_ext = np.zeros((len(group["u"]), g_e.shape[1]), dtype=complex)
            g_ext[n_pad:] = g_e
            lam = sim.Ainv[0] * g_ext
            dsig_col = -sim.getADeriv(
                freq, group["u"], lam.reshape(len(group["u"]), -1), adjoint=True
            )
            dsig_col = np.asarray(dsig_col).reshape(n_z, len(cols), n_rhs)
            np.add.at(dsigma, cells[cols].T, dsig_col)
        return dsigma[:, 0] if n_rhs == 1 else dsigma

    def getRHS(self, freq):
        r"""Right-hand sides for the given frequency.

        With ``boundary_condition="robin_1d"``, this includes the boundary data
        from the local 1D fields on the lateral boundaries. See the Notes
        section of :class:`Simulation3DPrimarySecondary`.

        Parameters
        ----------
        freq : float
            The frequency in Hz.

        Returns
        -------
        (n_edges, n_fields) numpy.ndarray
            The right-hand sides.
        """
        rhs = super().getRHS(freq)
        if self.boundary_condition == "robin_1d":
            rhs = np.array(rhs, dtype=complex).reshape(self.mesh.n_edges, -1)
            i = 0
            for src in self.survey.get_sources_by_frequency(freq):
                n = src._fields_per_source
                rhs[:, i : i + n] += self._robin_1d_boundary_data(freq, src)
                i += n
        return rhs

    def getRHSDeriv(self, freq, src, v, adjoint=False):
        r"""Derivative of the right-hand side times a vector for a given source.

        With ``boundary_condition="robin_1d"``, this includes the derivative of
        the boundary data from the local 1D fields.

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        src : .natural_source.sources.PlanewaveXYPrimary
            The source.
        v : numpy.ndarray
            The vector. (n_param,) for the standard operation. (n_edges, 2) for
            the adjoint operation.
        adjoint : bool
            Whether to perform the adjoint operation.

        Returns
        -------
        numpy.ndarray
            (n_edges, 2) for the standard operation. (n_param,) for the adjoint
            operation.
        """
        drhs = super().getRHSDeriv(freq, src, v, adjoint=adjoint)
        if self.boundary_condition != "robin_1d" or self.sigmaMap is None:
            return drhs

        if adjoint:
            dsigma = self._robin_1d_boundary_data_deriv(freq, src, v, adjoint=True)
            d_bnd = self.sigmaDeriv.T @ dsigma
        else:
            d_bnd = self._robin_1d_boundary_data_deriv(freq, src, self.sigmaDeriv @ v)
        if isinstance(drhs, Zero):
            return d_bnd
        return drhs + d_bnd.reshape(np.shape(drhs))
