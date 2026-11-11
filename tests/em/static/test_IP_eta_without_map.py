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
