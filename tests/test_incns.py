from __future__ import absolute_import, print_function, division
from six.moves import map, range
from os.path import abspath, basename, dirname, join
from firedrake import *
from firedrake.petsc import PETSc
from incflow import *
import pytest

cwd = abspath(dirname(__file__))
data_dir = join(cwd, "data")


@pytest.mark.skip(reason="regression test")
def test_incns_dfg_benchmark():
    mesh = Mesh(data_dir + "/cyl.e")

    dm = mesh._plex
    from firedrake.mg.impl import filter_exterior_facet_labels
    for _ in range(1):
        dm.setRefinementUniform(True)
        dm = dm.refine()
        dm.removeLabel("interior_facets")
        dm.removeLabel("op2_core")
        dm.removeLabel("op2_non_core")
        dm.removeLabel("op2_exec_halo")
        dm.removeLabel("op2_non_exec_halo")
        filter_exterior_facet_labels(dm)

    mesh = Mesh(dm, dim=mesh.ufl_cell().geometric_dimension(), distribute=False,
                reorder=True)

    incns = IncNavierStokes(mesh)

    W = incns.get_mixed_fs()

    x, y = SpatialCoordinate(mesh)
    inflow_profile = as_vector([4.0 * 1.5 * y * (0.41 - y) / 0.41**2, 0.0])

    bcu_inflow = DirichletBC(W.sub(0), Expression(inflow_profile, degree=2), (1,))
    bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), (2, 4))
    bcu_cylinder = DirichletBC(W.sub(0), Constant((0.0, 0.0)), (5,))
    bcp_outflow = DirichletBC(W.sub(1), Constant(0), (3,))

    incns.set_bcs(u_bcs=[bcu_walls, bcu_inflow, bcu_cylinder],
                  p_bcs=[bcp_outflow])

    incns.setup_solver()

    outfile = File(join(data_dir, "../", "results/", "test_incns.pvd"))

    step = 0
    t = 0.0
    t_end = 35.0

    while(t <= t_end):
        t += incns.dt

        printp0("***********************")
        printp0("Timestep {}".format(t))

        u1, p1 = incns.step()

        n = FacetNormal(mesh)
        D = -p1 * n[0] * ds(5)
        L = p1 * n[1] * ds(5)

        # Assemble functionals over sub domain
        drag = assemble(D)
        lift = assemble(L)

        print("C_L: {}".format((2.0 / 1.0**2 * 0.1) * lift))
        print("C_D: {}".format((2.0 / 1.0**2 * 0.1) * drag))

        if step % 50 == 0:
            outfile.write(u1, p1)

        step += 1


@pytest.mark.skip(reason="regression test")
def test_steady_mms():
    N = 32
    mesh = RectangleMesh(N, N, 1.0, 1.0)

    incns = IncNavierStokes(mesh)
    V = incns.get_u_fs()
    Q = incns.get_p_fs()

    mu = 1.0
    rho = 1.0
    incns.mu = 1.0
    incns.nu = 1.0
    incns.rho = 1.0
    incns.dt = 0.001

    x, y = SpatialCoordinate(mesh)
    u_analytical = as_vector([x**2 + y**2, 2.0 * x**2 - 2.0 * x * y])
    p_analytical = x + y - 1.0

    rho = Constant(rho)
    mu = Constant(mu)

    forcing = (
        rho * grad(u_analytical) * u_analytical
        - mu * rho * div(grad(u_analytical))
        + grad(p_analytical)
    )

    u_bcs = DirichletBC(V, u_analytical, (1, 2, 3, 4))
    p_bcs = DirichletBC(Q, p_analytical, (1, 2, 3, 4))
    incns.set_bcs(u_bcs=u_bcs, p_bcs=p_bcs)
    incns.set_forcing(forcing)
    incns.setup_solver()

    step = 0
    t = 0.0
    t_end = 1.0

    while(t <= t_end):
        t += incns.dt

        printp0("***********************")
        printp0("Timestep {}".format(t))

        u1, p1 = incns.step()

        printp0("")

        step += 1

    print(errornorm(interpolate(u_analytical, V), u1, norm_type="L2"))


if __name__ == "__main__":
    test_incns_dfg_benchmark()
#    test_steady_mms()
