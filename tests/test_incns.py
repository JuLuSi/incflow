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

    V = incns.get_u_fs()
    Q = incns.get_p_fs()

    inflow_profile = (
        "1.5 * 4.0 * x[1] * (0.41 - x[1]) / ( 0.41 * 0.41 )", "0.0")
    bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), (1,))
    bcu_walls = DirichletBC(V, Constant((0.0, 0.0)), (2, 4))
    bcu_cylinder = DirichletBC(V, Constant((0, 0)), (5,))
    bcp_outflow = DirichletBC(Q, Constant(0), (3,))

    # incns.set_bcs([bcu_inflow, bcu_walls, bcu_cylinder], [bcp_outflow])

    incns.set_bcs(u_bcs=[bcu_inflow, bcu_walls, bcu_cylinder],
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

        p_diff = p1.at([0.15, 0.2]) - p1.at([0.25, 0.2])
        printp0("p_diff = {}".format(p_diff))
        printp0("")

        if step % 50 == 0:
            outfile.write(u1, p1)

        step += 1


def test_steady_mms():
    N = 50
    mesh = RectangleMesh(N, N, 1.0, 1.0)

    incns = IncNavierStokes(mesh)
    V = incns.get_u_fs()
    Q = incns.get_p_fs()

    mu = 1.0
    nu = 1.0
    rho = 1.0
    incns.mu = 1.0
    incns.nu = 1.0
    incns.rho = 1.0
    incns.dt = 0.01

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

    vel_error = Function(V, name="velocity_error")

    step = 0
    t = 0.0
    t_end = 1.0

    outfile = File(join(data_dir, "../", "results/", "test_steady_mms.pvd"))

    while(t <= t_end):
        t += incns.dt

        printp0("***********************")
        printp0("Timestep {}".format(t))

        u1, p1 = incns.step()

        printp0("")
        vel_error.assign(interpolate(u_analytical, V) - u1)
        outfile.write(u1, p1, vel_error)

        step += 1

    print(errornorm(interpolate(u_analytical, V), u1, norm_type="L2", degree_rise=3))


# @pytest.mark.skip(reason="skip")
def test_unsteady_decaying_vortex2d():
    N = 100
    mesh = RectangleMesh(N, N, 2.0 * pi, 2.0 * pi)

    incns = IncNavierStokes(mesh)
    V = incns.get_u_fs()
    Q = incns.get_p_fs()

    mu = 0.01
    nu = 0.01
    rho = 1.0
    incns.mu = mu
    incns.nu = nu
    incns.rho = rho
    incns.dt = 0.01

    incns.time_integration_method = "crank_nicolson"

    x, y = SpatialCoordinate(mesh)

    t = Constant(0.0)

    F = exp(-2.0 * nu * t)
    u_analytical = as_vector([
        cos(x) * sin(y) * F,
        -sin(x) * cos(y) * F
    ])

    p_analytical = (-rho / 4.0 * (cos(2.0 * x) + cos(2.0 * y)) * F**2)

    forcing = Constant((0.0, 0.0))
    u_bcs = DirichletBC(V, u_analytical, (1, 2, 3, 4))
    p_bcs = DirichletBC(Q, p_analytical, (1, 2, 3, 4))
    incns.set_bcs(u_bcs=u_bcs, p_bcs=p_bcs)
    incns.set_forcing(forcing)
    incns.setup_solver()

    incns.u0.assign(project(u_analytical, V))
    incns.u_1.assign(project(u_analytical, V))
    incns.p0.assign(project(p_analytical, Q))

    pressure_error = Function(Q)
    pressure_error.rename("pressure_error")

    step = 0
    timestep = 0.0
    t_end = 0.1

    if True:
        outfile = File(join(data_dir, "../", "results/", "test_unsteady_decaying_vortex2d.pvd"))

    while(timestep <= t_end):
        timestep += incns.dt
        t.assign(timestep)

        printp0("***********************")
        printp0("Timestep {}".format(timestep))

        u1, p1 = incns.step()

        printp0("")

        if True:
            pressure_error.assign(interpolate(p_analytical, Q) - p1)
            outfile.write(u1, p1, pressure_error)

            print(errornorm(interpolate(u_analytical, V), u1, norm_type="L2", degree_rise=3))
            print(errornorm(interpolate(p_analytical, Q), p1, norm_type="L2", degree_rise=3))

        step += 1


def test_guermond():
    N = 50
    mesh = RectangleMesh(N, N, 1.0, 1.0)

    incns = IncNavierStokes(mesh)
    V = incns.get_u_fs()
    Q = incns.get_p_fs()

    mu = 0.01
    nu = 0.01
    rho = 1.0
    incns.mu = mu
    incns.nu = nu
    incns.rho = rho
    incns.dt = 0.01

    incns.time_integration_method = "backward_euler"

    x, y = SpatialCoordinate(mesh)

    t = Constant(0.0)
    t_sym = variable(t)

    u_analytical = as_vector([
        pi * sin(t_sym) * (sin(2.0 * pi * y) * sin(pi * x)**2),
        pi * sin(t_sym) * (-sin(2.0 * pi * x) * sin(pi * y)**2)
    ])

    p_analytical = sin(t_sym) * cos(pi * x) * sin(pi * y)

    forcing = (
        diff(u_analytical, t_sym) +
        rho * grad(u_analytical) * u_analytical
        - mu * rho * div(grad(u_analytical))
        + grad(p_analytical)
    )
    u_bcs = DirichletBC(V, u_analytical, (1, 2, 3, 4))
    p_bcs = DirichletBC(Q, p_analytical, (1, 2, 3, 4))
    incns.set_bcs(u_bcs=u_bcs, p_bcs=p_bcs)
    incns.set_forcing(forcing)
    incns.setup_solver()

    incns.u0.assign(project(u_analytical, V))
    incns.u_1.assign(project(u_analytical, V))
    incns.p0.assign(project(p_analytical, Q))

    pressure_error = Function(Q)
    pressure_error.rename("pressure_error")

    step = 0
    timestep = 0.0
    t_end = 0.1

    if True:
        outfile = File(join(data_dir, "../", "results/", "test_guermond.pvd"))

    while(timestep <= t_end):
        timestep += incns.dt
        t.assign(timestep)

        printp0("***********************")
        printp0("Timestep {}".format(timestep))

        u1, p1 = incns.step()

        printp0("")

        if True:
            pressure_error.assign(interpolate(p_analytical, Q) - p1)
            outfile.write(u1, p1, pressure_error)

            print(errornorm(interpolate(u_analytical, V), u1, norm_type="L2", degree_rise=3))
            print(errornorm(interpolate(p_analytical, Q), p1, norm_type="L2", degree_rise=3))

        step += 1


if __name__ == "__main__":
    # test_unsteady_decaying_vortex2d()
    pass
