from six.moves import map, range
from os.path import abspath, basename, dirname, join
from firedrake import *
from firedrake.petsc import PETSc
from incflow import *
import csv
import pytest

cwd = abspath(dirname(__file__))
data_dir = join(cwd, "data")


@pytest.mark.regression
def test_incns_dfg_benchmark():
    mesh = Mesh(data_dir + "/cyl.e")
    mesh = MeshHierarchy(mesh, 1)[-1]

    rho = 1.0
    nu = 0.001

    write_csv = True
    testcase = 2

    if testcase == 1:
        U_mean = 0.2
        U_inflow = 0.3
    elif testcase == 2:
        U_mean = 1.0
        U_inflow = 1.5
    elif testcase == 3:
        U_mean = 1.0
        tconst = Constant(0.0)
        U_inflow = 1.5 * sin(pi * tconst / 8.0)

    L_char = 0.1

    incns = IncNavierStokes(mesh, nu=nu, rho=rho)

    # incns.dt = 1.0 / 1600.0
    incns.dt = 0.005

    W = incns.get_mixed_fs()

    x, y = SpatialCoordinate(mesh)
    inflow_profile = as_vector([4.0 * U_inflow * y * (0.41 - y) / 0.41 ** 2, 0.0])

    bcu_inflow = DirichletBC(W.sub(0), Expression(inflow_profile, degree=2), (1,))
    bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), (2, 4))
    bcu_cylinder = DirichletBC(W.sub(0), Constant((0.0, 0.0)), (5,))
    bcp_outflow = DirichletBC(W.sub(1), Constant(0), (3,))

    incns.set_bcs(u_bcs=[bcu_walls, bcu_inflow, bcu_cylinder],
                  p_bcs=[bcp_outflow])

    incns.setup_solver()

    if testcase == 1:
        outfile = File(join(data_dir, "../", "results/", "dfg2d1.pvd"))
    elif testcase == 2:
        outfile = File(join(data_dir, "../", "results/", "dfg2d2.pvd"))

    step = 0
    t = 0.0
    if testcase == 1:
        t_end = 8.0
    elif testcase == 2:
        t_end = 30.0
    num_timesteps = int(t_end / incns.dt)
    output_frequency = 1

    print("Number of timesteps: {}".format(num_timesteps))
    print("Output frequency: {}".format(output_frequency))
    print("Number of output files: {}".format(int(num_timesteps / output_frequency)))
    print("DOFs: {}".format(incns.up0.vector().size()))

    # Lift / Drag
    n = FacetNormal(mesh)
    u1, p1 = incns.up.split()
    sigma = 2.0 * nu * sym(grad(u1)) - p1 * Identity(2)
    forces = -sigma * n

    if write_csv:
        if testcase == 1:
            csv_file = open('dfg2d1_points.csv', 'w', newline='')
        elif testcase == 2:
            csv_file = open('dfg2d2_points.csv', 'w', newline='')
        csv_file.write('time drag lift pdiff\n')

    while t <= t_end:
        t += incns.dt
        if testcase == 3:
            tconst.assign(t)
            bcu_inflow.set_value(U_inflow)

        printp0("***********************")
        printp0("Timestep {:.5f}".format(t))

        u1, p1 = incns.step()

        drag = assemble((2.0 / (U_mean ** 2 * L_char) * forces[0]) * ds(5))
        lift = assemble((2.0 / (U_mean ** 2 * L_char) * forces[1]) * ds(5))

        pp0 = p1.at(0.15, 0.2, tolerance=1e-12)
        pp1 = p1.at(0.25, 0.2, tolerance=1e-12)
        pdiff = pp0 - pp1

        print("C_D: {}".format(drag))
        print("C_L: {}".format(lift))
        print("p1: {}".format(pp0))
        print("p2: {}".format(pp1))
        print("pdiff: {}".format(pdiff))

        if write_csv:
            csv_file.write('{} {} {} {}\n'.format(t, drag, lift, pdiff))

        if step % 10 == 0:
            outfile.write(u1, p1, time=t)
            if write_csv:
                csv_file.flush()

        step += 1

    if write_csv:
        csv_file.close()


@pytest.mark.regression
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
    u_analytical = as_vector([x ** 2 + y ** 2, 2.0 * x ** 2 - 2.0 * x * y])
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

    while (t <= t_end):
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
