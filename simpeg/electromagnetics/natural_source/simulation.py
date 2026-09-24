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
from .utils.source_utils import (
    primary_e_1d_solution,
    _primary_e_1d_solution_and_deriv,
)
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

        return G.T.tocsr() @ MeMui @ G + 1j * omega(freq) * MfSigma

    def getADeriv_sigma(self, freq, u, v, adjoint=False):
        return 1j * omega(freq) * self.MfSigmaDeriv(u, v, adjoint=adjoint)

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        G = self.mesh.nodal_gradient
        if adjoint:
            return self.MeMuiDeriv(G * u, G * v, adjoint)
        return G.T * self.MeMuiDeriv(G * u, v, adjoint)

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

        return G.T.tocsr() @ MeRho @ G + 1j * omega(freq) * MnMu

    def getADeriv_rho(self, freq, u, v, adjoint=False):
        G = self.mesh.nodal_gradient
        if adjoint:
            return self.MeRhoDeriv(G * u, G * v, adjoint)
        return G.T * self.MeRhoDeriv(G * u, v, adjoint)

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        MnMuDeriv = self.MnMuDeriv(u)
        if adjoint is True:
            return 1j * omega(freq) * (MnMuDeriv.T * v)

        return 1j * omega(freq) * (MnMuDeriv * v)

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

        return C.T.tocsr() @ Mcc_mui @ C + 1j * omega(freq) * Me_sigma

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
        return 1j * omega(freq) * self.MeSigmaDeriv(u, v, adjoint=adjoint)

    def getADeriv_mui(self, freq, u, v, adjoint=False):
        C = self.mesh.edge_curl
        if adjoint:
            return self.MccMuiDeriv(C * u, C * v, adjoint)
        return C.T * self.MccMuiDeriv(C * u, v, adjoint)

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

        return C.T.tocsr() @ Mcc_rho @ C + 1j * omega(freq) * Me_mu

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
            return self.MccRhoDeriv(C * u, C * v, adjoint)
        return C.T * self.MccRhoDeriv(C * u, v, adjoint)

    def getADeriv_mu(self, freq, u, v, adjoint=False):
        return 1j * omega(freq) * self.MeMuDeriv(u, v, adjoint=adjoint)

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

    def _robin_1d_boundary_data(self, freq, src, deriv=False):
        """Right hand side boundary data from the local 1D boundary fields.

        Parameters
        ----------
        freq : float
            The frequency in Hz.
        src : .natural_source.sources.PlanewaveXYPrimary
            The source.
        deriv : bool
            Whether to also return the derivatives with respect to the
            conductivity of the cells.

        Returns
        -------
        rhs : (n_edges, 2) numpy.ndarray
            Boundary data for the x and y polarizations.
        drhs_dsigma : tuple of (n_edges, n_cells) scipy.sparse.csr_matrix
            Only returned if ``deriv`` is ``True``.
        """
        key = (freq, id(src))
        cache = getattr(self, "_robin_1d_cache", None)
        if cache is None:
            cache = self._robin_1d_cache = {}
        if key in cache and (not deriv or len(cache[key]) > 1):
            return cache[key] if deriv else cache[key][0]

        mesh = self.mesh
        geom = self._robin_1d_geometry
        w = omega(freq)
        hz = mesh.h[2]
        z_nodes = mesh.nodes_z

        sigma = self.sigma
        if np.size(sigma) == 1:
            sigma = np.full(mesh.n_cells, sigma)
        column_sigma = sigma[geom["column_cells"]]

        sigma_1d, _ = src._get_sigmas(self)
        e_p = primary_e_1d_solution(mesh, sigma_1d, freq)

        def h_top(e):
            # y component of h for an x-polarized field in the top cell
            return -(e[..., -1] - e[..., -2]) / hz[-1] / (1j * w * mu_0)

        unique_sigma, inverse = np.unique(column_sigma, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        e_d = np.empty((len(unique_sigma), len(z_nodes)), dtype=complex)
        de_d = np.empty((len(unique_sigma), len(z_nodes), len(hz)), dtype=complex)
        for i, sig in enumerate(unique_sigma):
            e_1d, de_1d = _primary_e_1d_solution_and_deriv(mesh, sig, freq)
            # use the same magnetic field at the top of the mesh as the primary
            scale = h_top(e_p) / h_top(e_1d)
            e_d[i] = scale * e_1d - e_p
            if deriv:
                de_d[i] = scale * (de_1d - np.outer(e_1d, h_top(de_1d.T) / h_top(e_1d)))

        col = inverse[geom["column"]]
        k_bot, k_top = geom["k_bot"], geom["k_top"]
        dz = z_nodes[k_top] - z_nodes[k_bot]
        is_vertical = geom["edge_type"] == 2
        dz[~is_vertical] = 1.0

        # E_d along horizontal edges and the y component of h_d (x-polarization)
        # on vertical edges; the y-polarization has h_d = -h_y x_hat.
        E = e_d[col, k_bot]
        Hy = -(e_d[col, k_top] - e_d[col, k_bot]) / dz / (1j * w * mu_0)
        admittivity = self._boundary_admittivity(freq)[geom["cell"]]
        a = np.sqrt(1j * w * admittivity / mu_0)
        weight = geom["weight"]
        normal = geom["normal"]

        rhs = np.zeros((mesh.n_edges, 2), dtype=complex)
        derivs = []
        for pol in range(2):
            is_tangent = geom["edge_type"] == pol
            n_comp = normal[:, 0] if pol == 0 else normal[:, 1]
            values = weight * (
                np.where(is_tangent, a * E, 0.0)
                + np.where(is_vertical, 1j * w * n_comp * Hy, 0.0)
            )
            rhs[:, pol] = np.bincount(
                geom["edge"], values.real, minlength=mesh.n_edges
            ) + 1j * np.bincount(geom["edge"], values.imag, minlength=mesh.n_edges)

            if deriv:
                cells = geom["column_cells"][geom["column"]]
                h, v = is_tangent, is_vertical
                # through the admittance of the adjacent cell
                rows = [geom["edge"][h]]
                cols = [geom["cell"][h]]
                vals = [weight[h] * 0.5 * a[h] / admittivity[h] * E[h]]
                # through the 1D field along the column
                dE = de_d[col[h], k_bot[h]]
                rows.append(np.repeat(geom["edge"][h], len(hz)))
                cols.append(cells[h].ravel())
                vals.append(((weight[h] * a[h])[:, None] * dE).ravel())
                dHy = -(de_d[col[v], k_top[v]] - de_d[col[v], k_bot[v]]) / (
                    dz[v, None] * mu_0
                )
                rows.append(np.repeat(geom["edge"][v], len(hz)))
                cols.append(cells[v].ravel())
                vals.append(((weight[v] * n_comp[v])[:, None] * dHy).ravel())
                derivs.append(
                    sp.csr_matrix(
                        (
                            np.concatenate(vals),
                            (np.concatenate(rows), np.concatenate(cols)),
                        ),
                        shape=(mesh.n_edges, mesh.n_cells),
                    )
                )

        cache[key] = (rhs, tuple(derivs)) if deriv else (rhs,)
        return cache[key] if deriv else rhs

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

        _, (d_x, d_y) = self._robin_1d_boundary_data(freq, src, deriv=True)
        if adjoint:
            v = np.asarray(v).reshape(self.mesh.n_edges, -1, 2)
            dsigma = d_x.T @ v[:, :, 0] + d_y.T @ v[:, :, 1]
            d_bnd = self.sigmaDeriv.T @ dsigma
            d_bnd = d_bnd[:, 0] if d_bnd.shape[1] == 1 else d_bnd
        else:
            dsigma = self.sigmaDeriv @ v
            d_bnd = np.column_stack([d_x @ dsigma, d_y @ dsigma])
        if isinstance(drhs, Zero):
            return d_bnd
        return drhs + d_bnd.reshape(np.shape(drhs))
