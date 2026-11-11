from functools import cached_property

import numpy as np
import scipy.sparse as sp

from .... import maps, props
from ....base import BasePDESimulation
from ....utils import Identity, Zero, mkvc
from ..resistivity import Simulation2DCellCentered as DC_2D_CC
from ..resistivity import Simulation2DNodal as DC_2D_N
from ..resistivity import Simulation3DCellCentered as DC_3D_CC
from ..resistivity import Simulation3DNodal as DC_3D_N


class BaseIPSimulation(BasePDESimulation):
    """Base class for linearized induced polarization simulations.

    The predicted data are the sensitivity of the DC data with respect to
    chargeability, multiplied by the chargeability. The chargeability can be given
    directly with `eta` (then call `dpred` without a model), or as a function of the
    inversion model with `etaMap`.
    """

    sigma = props.PhysicalProperty("Electrical Conductivity (S/m)")
    rho = props.PhysicalProperty("Electrical Resistivity (Ohm m)")
    props.Reciprocal(sigma, rho)

    @property
    def sigmaMap(self):
        return maps.IdentityMap()

    @sigmaMap.setter
    def sigmaMap(self, arg):
        pass

    @property
    def rhoMap(self):
        return maps.IdentityMap()

    @rhoMap.setter
    def rhoMap(self, arg):
        pass

    # The DC machinery (``getADeriv``, ``_Jtvec``, ...) works with respect to eta
    # through these two derivatives. The derivative of eta with respect to the
    # inversion model is applied separately in ``Jvec``, ``Jtvec`` and ``getJ``.
    @property
    def sigmaDeriv(self):
        return -sp.diags(self.sigma * np.ones(self.mesh.n_cells))

    @property
    def rhoDeriv(self):
        return sp.diags(self.rho * np.ones(self.mesh.n_cells))

    @property
    def _n_sensitivity_params(self):
        # The DC sensitivities are taken with respect to eta on every cell.
        return self.mesh.n_cells

    @cached_property
    def _scale(self):
        survey_slices = self.survey.get_all_slices()
        scale = np.ones(self.survey.nD)
        f = self.fields()
        try:
            f = self.fields_to_space(f)
        except AttributeError:
            pass
        # loop through receivers to check if they need to set the _dc_voltage
        for src in self.survey.source_list:
            for rx in src.receiver_list:
                if rx.data_type == "apparent_chargeability":
                    src_rx_slice = survey_slices[src, rx]
                    scale[src_rx_slice] = mkvc(1.0 / rx.eval(src, self.mesh, f))
        return scale

    eta, etaMap, etaDeriv = props.Invertible("Electrical Chargeability (V/V)")

    def __init__(
        self,
        mesh,
        survey=None,
        sigma=None,
        rho=None,
        eta=None,
        etaMap=None,
        Ainv=None,  # A DC's Ainv
        _f=None,  # A pre-computed DC field
        **kwargs,
    ):
        super().__init__(mesh=mesh, survey=survey, **kwargs)
        self.sigma = sigma
        self.rho = rho
        self.eta = eta
        self.etaMap = etaMap
        if Ainv is not None:
            self.Ainv = Ainv
        self._f = _f

    _f = None
    _Jmatrix = None

    # Everything cached here depends on the background conductivity (not on eta),
    # so it is kept across model updates, and cleared when `sigma` or `rho` is set.
    @property
    def _clear_on_sigma_update(self):
        return super()._clear_on_sigma_update + [
            "_f",
            "_Jmatrix",
            "_gtgdiag",
            "_scale",
        ]

    @property
    def _clear_on_rho_update(self):
        return super()._clear_on_rho_update + [
            "_f",
            "_Jmatrix",
            "_gtgdiag",
            "_scale",
        ]

    def fields(self, m=None):
        if m is not None:
            self.model = m
        # computed once, the DC fields only depend on the background conductivity
        if self._f is None:
            if self.verbose:
                print(">> Compute DC fields")
            # start from a new factorization, the DC fields method would otherwise
            # clean (or in 2D, overwrite) an `Ainv` that could be shared with the DC
            # simulation it came from.
            if isinstance(self.Ainv, list):
                self.Ainv = [None] * len(self.Ainv)
            else:
                self.Ainv = None
            self._f = super().fields(None)
        return self._f

    def dpred(self, m=None, f=None):
        if f is None:
            f = self.fields(m)
        return self.forward(m, f=f)

    def getJtJdiag(self, m, W=None, f=None):
        if getattr(self, "_gtgdiag", None) is None:
            J = self.getJ(m, f=f)
            if W is None:
                W = self._scale**2
            else:
                W = (self._scale * W.diagonal()) ** 2

            self._gtgdiag = np.einsum("i,ij,ij->j", W, J, J)

        return self._gtgdiag

    def _getJ_eta(self, f):
        # Sensitivity of the unscaled data with respect to eta (cached in `_Jmatrix`)
        return super().getJ(self.model, f=f)

    def _Jvec_eta(self, v, f):
        if self.storeJ:
            return np.asarray(self._getJ_eta(f) @ v)
        return super().Jvec(self.model, v, f=f)

    def _Jtvec_eta(self, v, f):
        if self.storeJ:
            return np.asarray(self._getJ_eta(f).T @ v)
        return super().Jtvec(self.model, v, f=f)

    def getJ(self, m, f=None):
        self.model = m
        if f is None:
            f = self.fields()
        J_eta = self._getJ_eta(f)
        eta_deriv = self.etaDeriv
        if isinstance(eta_deriv, Zero):
            return Zero()
        if isinstance(eta_deriv, Identity):
            return J_eta
        return (eta_deriv.T @ J_eta.T).T

    def Jvec(self, m, v, f=None):
        self.model = m
        eta_deriv_v = self.etaDeriv @ v
        if isinstance(eta_deriv_v, Zero):
            return np.zeros(self.survey.nD)
        if f is None:
            f = self.fields()
        return self._scale * self._Jvec_eta(eta_deriv_v, f)

    def forward(self, m, f=None):
        self.model = m
        if f is None:
            f = self.fields()
        eta = self.eta * np.ones(self.mesh.n_cells)
        return self._scale * self._Jvec_eta(eta, f)

    def Jtvec(self, m, v, f=None):
        self.model = m
        eta_deriv = self.etaDeriv
        if isinstance(eta_deriv, Zero):
            return Zero()
        if f is None:
            f = self.fields()
        return eta_deriv.T @ self._Jtvec_eta(v * self._scale, f)

    @property
    def _delete_on_model_update(self):
        toDelete = []
        return toDelete


class Simulation2DCellCentered(BaseIPSimulation, DC_2D_CC):
    """
    2.5D cell centered IP problem
    """


class Simulation2DNodal(BaseIPSimulation, DC_2D_N):
    """
    2.5D nodal IP problem
    """


class Simulation3DCellCentered(BaseIPSimulation, DC_3D_CC):
    """
    3D cell centered IP problem
    """


class Simulation3DNodal(BaseIPSimulation, DC_3D_N):
    """
    3D nodal IP problem
    """


Simulation2DCellCentred = Simulation2DCellCentered
Simulation3DCellCentred = Simulation3DCellCentered
