"""
Tests for the boundary conditions of the 3D primary-secondary NSEM simulation.
"""

import numpy as np
import pytest
import discretize
from discretize.tests import check_derivative
from scipy.constants import mu_0

from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem
from simpeg.electromagnetics.frequency_domain import Simulation3DElectricField
from simpeg.utils import get_default_solver


def get_tensor_mesh():
    h = [(100.0, 3, -1.5), (100.0, 6), (100.0, 3, 1.5)]
    return discretize.TensorMesh([h, h, h], "CCC")


def get_tree_mesh():
    mesh = discretize.TreeMesh(
        [np.full(16, 100.0)] * 3, origin="CCC", diagonal_balance=True
    )
    mesh.refine_ball([0, 0, 0], 300, levels=-1, finalize=False)
    mesh.refine(-2)
    return mesh


def get_simulation(mesh, **kwargs):
    survey = nsem.survey.Survey([nsem.sources.PlanewaveXYPrimary([], 1.0)])
    return nsem.simulation.Simulation3DPrimarySecondary(
        mesh, survey=survey, sigmaPrimary=1e-2, **kwargs
    )


def test_default_boundary_condition():
    sim = get_simulation(get_tensor_mesh(), sigma=1e-2)
    assert sim.boundary_condition == "robin"


def test_bad_boundary_condition():
    with pytest.raises(ValueError):
        get_simulation(get_tensor_mesh(), sigma=1e-2, boundary_condition="dirichlet")


def test_natural_matches_electric_field_simulation():
    mesh = get_tensor_mesh()
    sim = get_simulation(mesh, sigma=1e-2, boundary_condition="natural")
    sim_e = Simulation3DElectricField(mesh, survey=sim.survey, sigma=1e-2)
    diff = sim.getA(1.0) - sim_e.getA(1.0)
    assert diff.count_nonzero() == 0


@pytest.mark.parametrize("bc", ["robin", "robin_1d"])
def test_robin_only_on_lateral_and_bottom_boundary_edges(bc):
    mesh = get_tensor_mesh()
    sim = get_simulation(mesh, sigma=1e-2, boundary_condition=bc)
    sim_nat = get_simulation(mesh, sigma=1e-2, boundary_condition="natural")
    diff = (sim.getA(1.0) - sim_nat.getA(1.0)).tocoo()

    # the difference is diagonal
    np.testing.assert_equal(diff.row, diff.col)

    # and only on edges that lie on the x, y or bottom boundaries
    edges = np.r_[mesh.edges_x, mesh.edges_y, mesh.edges_z]
    on_robin = (
        np.isclose(edges[:, 0], mesh.nodes_x[0])
        | np.isclose(edges[:, 0], mesh.nodes_x[-1])
        | np.isclose(edges[:, 1], mesh.nodes_y[0])
        | np.isclose(edges[:, 1], mesh.nodes_y[-1])
        | np.isclose(edges[:, 2], mesh.nodes_z[0])
    )
    assert np.all(on_robin[diff.row])
    # edges only on the top boundary are untouched
    on_top_only = np.isclose(edges[:, 2], mesh.nodes_z[-1]) & ~on_robin
    assert not np.any(on_top_only[diff.row])


@pytest.mark.parametrize("sigma", [1e-3, 1e-1])
def test_robin_absorbs_outgoing_planewave(sigma):
    """A plane wave leaving through the +x face satisfies the Robin condition."""
    h = 25.0
    mesh = discretize.TensorMesh([np.full(64, h), np.full(8, h), np.full(8, h)])
    freq = 1.0
    w = 2 * np.pi * freq
    k = np.sqrt(-1j * w * mu_0 * sigma)

    # y-polarized plane wave travelling in the +x direction
    e = np.r_[
        np.zeros(mesh.n_edges_x),
        np.exp(-1j * k * mesh.edges_y[:, 0]),
        np.zeros(mesh.n_edges_z),
    ]

    # y-edges on the +x face, away from the top and bottom boundaries
    edges_y = mesh.edges_y
    on_face = (
        np.isclose(edges_y[:, 0], mesh.nodes_x[-1])
        & (edges_y[:, 2] > mesh.nodes_z[0])
        & (edges_y[:, 2] < mesh.nodes_z[-1])
    )
    inds = mesh.n_edges_x + np.where(on_face)[0]

    sim = get_simulation(mesh, sigma=sigma)
    reference = np.abs(1j * w * (sim.MeSigma @ e)[inds])

    residual_robin = np.abs((sim.getA(freq) @ e)[inds])
    sim.boundary_condition = "natural"
    residual_natural = np.abs((sim.getA(freq) @ e)[inds])

    assert np.all(residual_natural > reference)
    np.testing.assert_array_less(residual_robin, 1e-2 * reference)


@pytest.mark.parametrize("mesh_type", ["tensor", "tree"])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_robin_system_matrix_derivative(mesh_type, n_pol):
    mesh = get_tensor_mesh() if mesh_type == "tensor" else get_tree_mesh()
    active = mesh.cell_centers[:, 2] < 0
    mapping = maps.InjectActiveCells(mesh, active, 1e-8) * maps.ExpMap(nP=active.sum())
    sim = get_simulation(mesh, sigmaMap=mapping)

    rng = np.random.default_rng(542)
    m0 = np.log(1e-2) + rng.normal(scale=0.5, size=active.sum())
    shape = (mesh.n_edges, n_pol) if n_pol > 1 else (mesh.n_edges,)
    u = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    freq = 1.0

    def func(m):
        sim.model = m
        return sim.getA(freq) @ u, lambda v: sim.getADeriv(freq, u, v)

    assert check_derivative(func, m0, plotIt=False, num=3, random_seed=21)

    # the adjoint operation is the (non-conjugated) transpose
    sim.model = m0
    v = rng.normal(size=m0.shape)
    w = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    np.testing.assert_allclose(
        np.sum(w * sim.getADeriv(freq, u, v)),
        np.sum(v * sim.getADeriv(freq, u, w, adjoint=True)),
        rtol=1e-10,
    )


def test_robin_layered_earth_with_halfspace_primary():
    """Conductivity at the lateral boundaries differs from the primary model.

    The natural boundary condition requires the secondary field to vanish at
    the boundary, which cannot happen here, while the Robin boundary condition
    gives a much more accurate result.
    """
    frequencies = [0.1, 1.0]
    sigma_layers = np.r_[1e-2, 1e-1, 1e-2]
    thicknesses = np.r_[500.0, 1000.0]

    survey_1d = nsem.survey.Survey(
        [
            nsem.sources.Planewave(
                [nsem.receivers.Impedance([[]], orientation="xy", component="app_res")],
                f,
            )
            for f in frequencies
        ]
    )
    # the 1D simulation orders its layers from the bottom up
    sim_1d = nsem.simulation_1d.Simulation1DRecursive(
        survey=survey_1d,
        sigmaMap=maps.IdentityMap(),
        thicknesses=thicknesses[::-1],
    )
    app_res_1d = sim_1d.dpred(sigma_layers[::-1])

    hx = [(250.0, 4, -1.5), (250.0, 6), (250.0, 4, 1.5)]
    hz = [(250.0, 10, -1.4), (100.0, 20), (250.0, 10, 1.5)]
    mesh = discretize.TensorMesh([hx, hx, hz], "CC0")
    mesh.origin = mesh.origin - np.r_[0, 0, mesh.h[2][:30].sum()]

    zc = mesh.cell_centers[:, 2]
    sigma = np.full(mesh.n_cells, 1e-8)
    sigma[zc < 0] = sigma_layers[0]
    sigma[(zc < -thicknesses[0]) & (zc > -thicknesses.sum())] = sigma_layers[1]
    sigma_primary = np.where(zc < 0, sigma_layers[0], 1e-8)

    receivers = [
        nsem.receivers.Impedance(
            np.array([[0.0, 0.0, 0.0]]), orientation=orientation, component="app_res"
        )
        for orientation in ["xy", "yx"]
    ]
    survey = nsem.survey.Survey(
        [nsem.sources.PlanewaveXYPrimary(receivers, f) for f in frequencies]
    )

    errors = {}
    for bc in ["natural", "robin"]:
        sim = nsem.simulation.Simulation3DPrimarySecondary(
            mesh,
            survey=survey,
            sigma=sigma,
            sigmaPrimary=sigma_primary,
            boundary_condition=bc,
            solver=get_default_solver(),
        )
        app_res = sim.dpred().reshape(len(frequencies), 2)
        errors[bc] = np.abs(app_res / app_res_1d[:, None] - 1)

    np.testing.assert_array_less(errors["robin"], errors["natural"])
    np.testing.assert_array_less(errors["robin"], 0.25)


def layered_earth_mesh(npad):
    hx = [(250.0, npad, -1.5), (250.0, 6), (250.0, npad, 1.5)]
    hz = [(250.0, 10, -1.4), (100.0, 20), (250.0, 10, 1.5)]
    mesh = discretize.TensorMesh([hx, hx, hz], "CC0")
    mesh.origin = mesh.origin - np.r_[0, 0, mesh.h[2][:30].sum()]
    return mesh


def test_robin_1d_exact_for_laterally_uniform_boundaries():
    """robin_1d reproduces the matched-primary solution with a wrong primary.

    For a layered earth, the natural boundary condition with the exact layered
    primary gives zero secondary field. With a halfspace primary instead, the
    local 1D boundary data make the result identical to that solution, even
    with very little padding.
    """
    mesh = layered_earth_mesh(npad=2)
    zc = mesh.cell_centers[:, 2]
    sigma = np.full(mesh.n_cells, 1e-8)
    sigma[zc < 0] = 1e-2
    sigma[(zc < -500) & (zc > -1500)] = 1e-1
    sigma_halfspace = np.where(zc < 0, 1e-2, 1e-8)

    receivers = [
        nsem.receivers.Impedance(
            np.array([[17.0, 13.0, 0.0]]), orientation=orientation, component=comp
        )
        for orientation in ["xy", "yx"]
        for comp in ["real", "imag"]
    ]

    def dpred(sigma_primary, bc):
        survey = nsem.survey.Survey(
            [nsem.sources.PlanewaveXYPrimary(receivers, f) for f in [0.1, 10.0]]
        )
        sim = nsem.simulation.Simulation3DPrimarySecondary(
            mesh,
            survey=survey,
            sigma=sigma,
            sigmaPrimary=sigma_primary,
            boundary_condition=bc,
            solver=get_default_solver(),
        )
        return sim.dpred()

    reference = dpred(sigma, "natural")
    np.testing.assert_allclose(dpred(sigma_halfspace, "robin_1d"), reference, rtol=1e-6)
    # without the boundary data, the result is not accurate on this small mesh
    assert np.max(np.abs(dpred(sigma_halfspace, "robin") / reference - 1)) > 0.1


@pytest.mark.parametrize("mesh_type", ["tensor", "tree"])
def test_robin_1d_derivatives(mesh_type):
    mesh = get_tensor_mesh() if mesh_type == "tensor" else get_tree_mesh()
    zc = mesh.cell_centers[:, 2]
    active = zc < 0
    mapping = maps.InjectActiveCells(mesh, active, 1e-8) * maps.ExpMap(nP=active.sum())
    rng = np.random.default_rng(2025)
    # heterogeneous model, including the boundary columns
    m0 = np.log(1e-2) + rng.normal(scale=0.5, size=active.sum())

    locs = np.c_[np.linspace(-200, 200, 3) + 7, np.full(3, 11.0), np.full(3, -1.0)]
    receivers = [
        nsem.receivers.Impedance(locs, orientation=o, component=c)
        for o in ["xy", "yx", "xx"]
        for c in ["real", "imag"]
    ]
    receivers.append(nsem.receivers.Tipper(locs, orientation="zx", component="real"))
    # two sources at the same frequency with different primary models
    sources = [
        nsem.sources.PlanewaveXYPrimary(
            receivers, 1.0, sigma_primary=np.where(active, 1e-2, 1e-8)
        ),
        nsem.sources.PlanewaveXYPrimary(
            receivers, 1.0, sigma_primary=np.where(active, 3e-2, 1e-8)
        ),
    ]
    sim = nsem.simulation.Simulation3DPrimarySecondary(
        mesh,
        survey=nsem.survey.Survey(sources),
        sigmaMap=mapping,
        boundary_condition="robin_1d",
        solver=get_default_solver(),
    )

    # right hand side derivative for the second source
    def rhs(m):
        sim.model = m
        return sim.getRHS(1.0)[:, 2:4], lambda v: sim.getRHSDeriv(1.0, sources[1], v)

    assert check_derivative(rhs, m0, plotIt=False, num=3, random_seed=41)

    sim.model = m0
    v = rng.normal(size=m0.shape)
    w = rng.normal(size=(mesh.n_edges, 2)) + 1j * rng.normal(size=(mesh.n_edges, 2))
    np.testing.assert_allclose(
        np.sum(w * sim.getRHSDeriv(1.0, sources[1], v)),
        np.sum(v * sim.getRHSDeriv(1.0, sources[1], w, adjoint=True)),
        rtol=1e-10,
    )

    # full sensitivities
    def dpred(m):
        return sim.dpred(m), lambda v: sim.Jvec(m, v)

    assert check_derivative(dpred, m0, plotIt=False, num=3, random_seed=42)

    f = sim.fields(m0)
    d = rng.normal(size=sim.survey.nD)
    np.testing.assert_allclose(
        d @ sim.Jvec(m0, v, f=f), v @ sim.Jtvec(m0, d, f=f), rtol=1e-10
    )
