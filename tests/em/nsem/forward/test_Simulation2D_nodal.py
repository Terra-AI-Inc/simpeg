"""
Tests for the nodal 2D NSEM simulation, Simulation2DNodalElectricField.
"""

import numpy as np
import pytest
import discretize

from simpeg import maps, tests
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import get_default_solver

FREQUENCIES = [0.1, 1.0, 10.0]


def layered(z):
    sigma = np.full(z.shape, 1e-8)
    sigma[z < 0] = 1e-1
    sigma[z < -3000] = 1e-2
    return sigma


def contact(cell_centers):
    x, z = cell_centers.T
    sigma = np.full(len(x), 1e-8)
    sigma[(z < 0) & (x < 0)] = 1e-1
    sigma[(z < 0) & (x >= 0)] = 1e-3
    sigma[z < -3000] = 1e-2
    return sigma


def make_survey(locations, receivers):
    receiver_list = []
    for rx_class, orientation, component in receivers:
        receiver_list.append(
            rx_class(locations, orientation=orientation, component=component)
        )
    return nsem.Survey([nsem.sources.Planewave(receiver_list, f) for f in FREQUENCIES])


IMPEDANCE = [
    (nsem.receivers.Impedance, "yx", "app_res"),
    (nsem.receivers.Impedance, "yx", "phase"),
]
TIPPER = [
    (nsem.receivers.Tipper, "zx", "real"),
    (nsem.receivers.Tipper, "zx", "imag"),
]


def padded_mesh(dx, core_x=2000.0, core_z=4000.0, distance=100e3):
    """Uniform core cells, padded out to a fixed distance on every side."""
    width, n_pad, remaining = dx, 0, distance
    while remaining > 0:
        width *= 1.3
        remaining -= width
        n_pad += 1
    n_x, n_z = int(2 * core_x / dx), int(core_z / dx)
    hx = [(dx, n_pad, -1.3), (dx, n_x), (dx, n_pad, 1.3)]
    hz = [(dx, n_pad, -1.3), (dx, n_z), (dx, n_pad, 1.3)]
    mesh = discretize.TensorMesh([hx, hz], "C0")
    mesh.origin = np.r_[mesh.origin[0], -mesh.h[1][: n_pad + n_z].sum()]
    return mesh


def test_layered_matches_1d():
    """For a layered earth it reproduces the 1D simulation's discretization."""
    h = [(50.0, 12, -1.3), (50.0, 80), (50.0, 12, 1.3)]
    mesh = discretize.TensorMesh([h, h], "CC")
    mesh.origin = mesh.origin + np.r_[0, -mesh.origin[1] - mesh.h[1][:52].sum()]
    locations = np.array([[-500.0, 0.0], [0.0, 0.0], [700.0, 0.0]])
    sim = nsem.Simulation2DNodalElectricField(
        mesh,
        survey=make_survey(locations, IMPEDANCE + TIPPER),
        sigma=layered(mesh.cell_centers[:, 1]),
        solver=get_default_solver(),
    )
    d_2d = sim.dpred().reshape(len(FREQUENCIES), 4, len(locations))

    mesh_1d = discretize.TensorMesh([mesh.h[1]], origin=[mesh.nodes_y[0]])
    sim_1d = nsem.Simulation1DElectricField(
        mesh_1d,
        survey=make_survey(np.array([[0.0]]), IMPEDANCE),
        sigma=layered(mesh_1d.cell_centers),
        solver=get_default_solver(),
    )
    d_1d = sim_1d.dpred().reshape(len(FREQUENCIES), 2, 1)
    np.testing.assert_allclose(d_2d[:, :2], np.broadcast_to(d_1d, (3, 2, 3)))
    np.testing.assert_allclose(d_2d[:, 2:], 0, atol=1e-10)


def test_converges_with_magnetic_field_simulation():
    """It solves the same mode as Simulation2DMagneticField, on the dual grid.

    Across a 100:1 contact, the difference between the two halves when the
    cells are halved.
    """
    locations = np.c_[np.linspace(-1500, 1500, 7), np.zeros(7)]
    receivers = [(nsem.receivers.Impedance, "yx", "app_res")]
    diffs = []
    for dx in (100.0, 50.0):
        mesh = padded_mesh(dx)
        sigma = contact(mesh.cell_centers)
        d = []
        for sim_class in (
            nsem.Simulation2DNodalElectricField,
            nsem.Simulation2DMagneticField,
        ):
            sim = sim_class(
                mesh,
                survey=make_survey(locations, receivers),
                sigma=sigma,
                solver=get_default_solver(),
            )
            d.append(sim.dpred())
        diffs.append(np.max(np.abs(d[0] - d[1]) / d[1]))
    assert diffs[1] < 0.6 * diffs[0]
    assert diffs[1] < 0.1


def test_tree_mesh_matches_tensor_mesh():
    h = [np.full(16, 100.0), np.full(64, 100.0)]
    origin = [-800.0, -4000.0]
    tensor = discretize.TensorMesh(h, origin=origin)
    tree = discretize.TreeMesh(h, origin=origin, diagonal_balance=True)
    tree.refine(-1)
    locations = np.array([[-350.0, 0.0], [50.0, 0.0], [450.0, 0.0]])
    d = []
    for mesh in (tensor, tree):
        sim = nsem.Simulation2DNodalElectricField(
            mesh,
            survey=make_survey(locations, IMPEDANCE + TIPPER),
            sigma=contact(mesh.cell_centers),
            solver=get_default_solver(),
        )
        d.append(sim.dpred())
    np.testing.assert_allclose(d[1], d[0], rtol=1e-8, atol=1e-12)


def refined_tree():
    h = [np.full(16, 100.0), np.full(64, 100.0)]
    mesh = discretize.TreeMesh(h, origin=[-800.0, -4000.0], diagonal_balance=True)
    mesh.refine_box([-800, -1500], [800, 0], -1, finalize=False)
    mesh.refine(mesh.max_level - 2, finalize=True)
    return mesh


@pytest.mark.parametrize("mesh_type", ["tensor", "tree"])
def test_derivatives(mesh_type):
    if mesh_type == "tensor":
        mesh = discretize.TensorMesh(
            [np.full(16, 100.0), np.full(64, 100.0)], origin=[-800.0, -4000.0]
        )
    else:
        # hanging nodes, and coarse cells along the sides and bottom
        mesh = refined_tree()
    locations = np.array([[-350.0, 0.0], [50.0, 0.0], [450.0, 0.0]])
    sim = nsem.Simulation2DNodalElectricField(
        mesh,
        survey=make_survey(locations, IMPEDANCE + TIPPER),
        sigmaMap=maps.ExpMap(mesh),
        solver=get_default_solver(),
    )
    m0 = np.log(contact(mesh.cell_centers))

    def fun(m):
        return sim.dpred(m), lambda v: sim.Jvec(m, v)

    assert tests.check_derivative(fun, m0, num=3, plotIt=False, random_seed=42)

    rng = np.random.default_rng(42)
    v = rng.normal(size=mesh.n_cells)
    w = rng.normal(size=sim.survey.nD)
    f = sim.fields(m0)
    vJw = v @ sim.Jtvec(m0, w, f=f)
    wJv = w @ sim.Jvec(m0, v, f=f)
    np.testing.assert_allclose(vJw, wJv, rtol=1e-8)


def test_invalid_receiver():
    mesh = discretize.TensorMesh([np.full(8, 100.0), np.full(8, 100.0)], "CC")
    survey = make_survey(
        np.array([[0.0, 0.0]]), [(nsem.receivers.Impedance, "xy", "app_res")]
    )
    with pytest.raises(TypeError, match="Simulation2DNodalElectricField"):
        nsem.Simulation2DNodalElectricField(mesh, survey=survey)
