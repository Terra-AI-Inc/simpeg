"""
Tests for IP simulations given ``eta`` directly, and with a non-linear ``etaMap``.
"""

import discretize
import numpy as np
import pytest

from simpeg import maps, tests
from simpeg.electromagnetics import induced_polarization as ip
from simpeg.electromagnetics import resistivity as dc

SIMULATIONS = [
    ip.Simulation3DNodal,
    ip.Simulation3DCellCentered,
    ip.Simulation2DNodal,
    ip.Simulation2DCellCentered,
]


def _is_2d(sim_class):
    return sim_class in (ip.Simulation2DNodal, ip.Simulation2DCellCentered)


def build_problem(sim_class, data_type="volt"):
    """Small mesh, survey and chargeability block for ``sim_class``."""
    if _is_2d(sim_class):
        mesh = discretize.TensorMesh(
            [
                [(10.0, 5, -1.3), (10.0, 20), (10.0, 5, 1.3)],
                [(10.0, 5, -1.3), (10.0, 10)],
            ],
            "CN",
        )
        x = np.linspace(-60.0, 60.0, 7)
        M = np.c_[x - 10.0, np.zeros_like(x)]
        N = np.c_[x + 10.0, np.zeros_like(x)]
        A, B = np.r_[-90.0, 0.0], np.r_[90.0, 0.0]
        center = np.r_[0.0, -40.0]
    else:
        mesh = discretize.TensorMesh(
            [[(10.0, 3, -1.3), (10.0, 14), (10.0, 3, 1.3)]] * 2
            + [[(10.0, 3, -1.3), (10.0, 8)]],
            "CCN",
        )
        x = np.linspace(-40.0, 40.0, 5)
        M = np.c_[x - 10.0, np.zeros_like(x), np.zeros_like(x)]
        N = np.c_[x + 10.0, np.zeros_like(x), np.zeros_like(x)]
        A, B = np.r_[-70.0, 0.0, 0.0], np.r_[70.0, 0.0, 0.0]
        center = np.r_[0.0, 0.0, -40.0]

    rx = dc.receivers.Dipole(M, N, data_type=data_type)
    src = dc.sources.Dipole([rx], A, B)
    survey = ip.Survey([src])

    sigma = np.full(mesh.n_cells, 1e-2)
    in_block = np.linalg.norm(mesh.cell_centers - center, axis=1) < 25.0
    eta = np.full(mesh.n_cells, 0.01)
    eta[in_block] = 0.1
    return mesh, survey, sigma, eta


@pytest.mark.parametrize("sim_class", SIMULATIONS)
@pytest.mark.parametrize("data_type", ["volt", "apparent_chargeability"])
@pytest.mark.parametrize("store_J", [False, True])
def test_eta_only_matches_mapped(sim_class, data_type, store_J):
    mesh, survey, sigma, eta = build_problem(sim_class, data_type)

    sim_map = sim_class(
        mesh, survey=survey, sigma=sigma, etaMap=maps.IdentityMap(), storeJ=store_J
    )
    sim_eta = sim_class(mesh, survey=survey, sigma=sigma, eta=eta, storeJ=store_J)

    d_map = sim_map.dpred(eta)
    d_eta = sim_eta.dpred()
    assert np.any(d_map != 0)
    np.testing.assert_allclose(d_eta, d_map, rtol=1e-10, atol=0)


@pytest.mark.parametrize("sim_class", SIMULATIONS)
def test_homogeneous_eta_apparent_chargeability(sim_class):
    """A uniform (scalar) eta gives an apparent chargeability equal to eta."""
    mesh, survey, sigma, _ = build_problem(sim_class, "apparent_chargeability")
    sim = sim_class(mesh, survey=survey, sigma=sigma, eta=0.05)
    np.testing.assert_allclose(sim.dpred(), 0.05, rtol=1e-6)


@pytest.mark.parametrize("sim_class", SIMULATIONS)
def test_nonlinear_map_forward(sim_class):
    """dpred(m) with an ExpMap is the IP response of eta = exp(m)."""
    mesh, survey, sigma, eta = build_problem(sim_class)

    sim_exp = sim_class(mesh, survey=survey, sigma=sigma, etaMap=maps.ExpMap())
    sim_eta = sim_class(mesh, survey=survey, sigma=sigma, eta=eta)

    np.testing.assert_allclose(
        sim_exp.dpred(np.log(eta)), sim_eta.dpred(), rtol=1e-10, atol=0
    )


@pytest.mark.parametrize("sim_class", SIMULATIONS)
@pytest.mark.parametrize("store_J", [False, True])
def test_nonlinear_map_derivatives(sim_class, store_J):
    mesh, survey, sigma, eta = build_problem(sim_class)
    sim = sim_class(
        mesh, survey=survey, sigma=sigma, etaMap=maps.ExpMap(), storeJ=store_J
    )
    m0 = np.log(eta)

    assert tests.check_derivative(
        lambda m: [sim.dpred(m), lambda dm: sim.Jvec(m, dm)],
        m0,
        plotIt=False,
        num=3,
        random_seed=42,
    )

    rng = np.random.default_rng(seed=7)
    v = rng.uniform(size=m0.size)
    w = rng.uniform(size=survey.nD)
    wtJv = w @ sim.Jvec(m0, v)
    vtJtw = v @ sim.Jtvec(m0, w)
    np.testing.assert_allclose(wtJv, vtJtw, rtol=1e-10)

    # The full sensitivity matrix agrees with the matrix-free products.
    J = sim.getJ(m0)
    np.testing.assert_allclose(J @ v, sim.Jvec(m0, v), rtol=1e-8)


@pytest.mark.parametrize("sim_class", SIMULATIONS)
def test_model_without_map_raises(sim_class):
    mesh, survey, sigma, eta = build_problem(sim_class)
    sim = sim_class(mesh, survey=survey, sigma=sigma, eta=eta)
    with pytest.raises(ValueError, match="no active mappings"):
        sim.dpred(eta)


@pytest.mark.parametrize("sim_class", SIMULATIONS)
def test_dpred_with_fields_uses_given_model(sim_class):
    """``dpred(m, f=f)`` predicts data for ``m``, whatever model built ``f``."""
    mesh, survey, sigma, eta = build_problem(sim_class)
    sim = sim_class(mesh, survey=survey, sigma=sigma, etaMap=maps.IdentityMap())

    f = sim.fields(eta)
    d_2eta = sim.dpred(2 * eta, f=f)
    np.testing.assert_allclose(d_2eta, 2 * sim.dpred(eta, f=f), rtol=1e-10)
    np.testing.assert_allclose(d_2eta, sim.dpred(2 * eta), rtol=1e-10)


@pytest.mark.parametrize("sim_class", SIMULATIONS)
@pytest.mark.parametrize("store_J", [False, True])
@pytest.mark.parametrize("prop", ["sigma", "rho"])
def test_changing_conductivity_clears_cache(sim_class, store_J, prop):
    """Setting `sigma` or `rho` after a run matches a new simulation."""
    mesh, survey, sigma, eta = build_problem(sim_class, "apparent_chargeability")
    sigma_new = sigma.copy()
    sigma_new[: mesh.n_cells // 2] *= 10.0

    sim = sim_class(mesh, survey=survey, sigma=sigma, eta=eta, storeJ=store_J)
    d_old = sim.dpred()
    setattr(sim, prop, sigma_new if prop == "sigma" else 1.0 / sigma_new)
    d_new = sim.dpred()

    sim_fresh = sim_class(mesh, survey=survey, sigma=sigma_new, eta=eta)
    assert not np.allclose(d_new, d_old)
    np.testing.assert_allclose(d_new, sim_fresh.dpred(), rtol=1e-8)

    # and the sensitivities of a mapped simulation
    sim = sim_class(
        mesh, survey=survey, sigma=sigma, etaMap=maps.ExpMap(), storeJ=store_J
    )
    m0 = np.log(eta)
    v = np.random.default_rng(seed=3).uniform(size=m0.size)
    sim.Jvec(m0, v)
    sim.getJtJdiag(m0)
    setattr(sim, prop, sigma_new if prop == "sigma" else 1.0 / sigma_new)

    sim_fresh = sim_class(
        mesh, survey=survey, sigma=sigma_new, etaMap=maps.ExpMap(), storeJ=store_J
    )
    np.testing.assert_allclose(sim.Jvec(m0, v), sim_fresh.Jvec(m0, v), rtol=1e-8)
    np.testing.assert_allclose(sim.getJtJdiag(m0), sim_fresh.getJtJdiag(m0), rtol=1e-8)


@pytest.mark.parametrize("sim_class", SIMULATIONS)
def test_shared_dc_solver_untouched(sim_class):
    """Changing the IP conductivity leaves a DC simulation's shared `Ainv` alone."""
    mesh, survey, sigma, eta = build_problem(sim_class)
    dc_class = getattr(dc, sim_class.__name__)
    dc_survey = dc.Survey(survey.source_list)
    sim_dc = dc_class(mesh, survey=dc_survey, sigmaMap=maps.ExpMap())
    m_dc = np.log(sigma)
    f_dc = sim_dc.fields(m_dc)
    v = np.random.default_rng(seed=5).uniform(size=m_dc.size)
    Jv_dc = sim_dc.Jvec(m_dc, v, f=f_dc)

    # without `_f`, the IP simulation computes its own DC fields
    sim_ip = sim_class(mesh, survey=survey, sigma=sigma, eta=eta, Ainv=sim_dc.Ainv)
    sim_ip.dpred()
    sim_ip.sigma = 10 * sigma
    sim_ip.dpred()

    np.testing.assert_allclose(sim_dc.Jvec(m_dc, v, f=f_dc), Jv_dc, rtol=1e-10)
