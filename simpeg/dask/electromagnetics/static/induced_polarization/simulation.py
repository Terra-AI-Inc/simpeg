import dask.array as da

from .....electromagnetics.static.induced_polarization.simulation import (
    BaseIPSimulation as Sim,
)
from .....utils import Identity, Zero


def dask_getJ(self, m, f=None):
    """
    Generate Full sensitivity matrix
    """
    self.model = m
    if f is None:
        f = self._dc_fields()
    J_eta = self._getJ_eta(f)
    eta_deriv = self.etaDeriv
    if isinstance(eta_deriv, Zero):
        return Zero()
    if isinstance(eta_deriv, Identity):
        return J_eta
    if isinstance(J_eta, da.Array):
        # keeps J lazy, ``J_eta @ eta_deriv`` isn't supported for sparse matrices
        return J_eta.dot(eta_deriv)
    return (eta_deriv.T @ J_eta.T).T


Sim.getJ = dask_getJ


def dask_getJtJdiag(self, m, W=None, f=None):
    """
    Return the diagonal of JtJ
    """
    if getattr(self, "_gtgdiag", None) is None:
        J = self.getJ(m, f=f)
        # Need to check if multiplying weights makes sense
        if W is None:
            W = self._scale
        else:
            W = self._scale * W.diagonal()
        self._gtgdiag = da.einsum("i,ij,ij->j", W**2, J, J).compute()

    return self._gtgdiag


Sim.getJtJdiag = dask_getJtJdiag
