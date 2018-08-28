from os.path import abspath, basename, dirname, join
from firedrake import *
from incflow import *
import pytest

set_log_level(DEBUG)

cwd = abspath(dirname(__file__))
data_dir = join(cwd, "data")


def test_temperature_driven_cavity():
    H = 8.0
    L = 1.0
    Pr = 0.71
    Ra = 3.4e5
    mu = Pr
    k = 1.0
    rho = 1.0
    cp = 1.0
    g = 1.0

    N = 128
    mesh = RectangleMesh(N / 4, N, L, H)
    x, y = SpatialCoordinate(mesh)

    write_csv = True

    incns = IncNavierStokes(mesh, nu=sqrt(Pr / Ra), rho=rho)
    eneq = EnergyEq(mesh, rho=rho, k=1.0 / sqrt(Ra * Pr), cp=cp)

    incns.dt = 0.001
    eneq.dt = incns.dt

    W = incns.get_mixed_fs()
    S = eneq.get_fs()

    T_cold = Constant(-0.5)
    T_hot = Constant(+0.5)

    bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), (1, 2, 3, 4))

    bcT = [
        DirichletBC(S, T_cold, (2,)),
        DirichletBC(S, T_hot, (1,))
    ]

    incns.set_bcs(u_bcs=[bcu_walls],
                  p_bcs=[])

    eneq.set_bcs(bcT)

    incns.has_nullspace = True

    incns.setup_solver()
    eneq.setup_solver(W.sub(0))
    outfile = File(join(data_dir, "../", "results/", "test_temp_driven_cavity.pvd"))

    eneq.T0.project(Constant(0.0))

    step = 0
    t = 0.0
    t_end = 1000.0
    num_timesteps = int(t_end / incns.dt)
    output_frequency = 1000

    print("Number of timesteps: {}".format(num_timesteps))
    print("Output frequency: {}".format(output_frequency))
    print("Number of output files: {}".format(int(num_timesteps / output_frequency)))
    print("INS DOFs: {}".format(incns.up0.vector().size()))
    print("ENEq DOFs: {}".format(eneq.T0.vector().size()))

    if write_csv:
        csv_file = open('mit81_points.csv', 'w', newline='')
        csv_file.write('time Tp1 up1 vp1 Nu0\n')

    n = FacetNormal(mesh)
    nusselt = dot(grad(eneq.T1), n)

    while t <= t_end:
        with Timer('Step'):
            t += incns.dt

            incns.set_forcing(
                eneq.T0 * as_vector([Constant(0.0), Constant(1.0)])
            )

            printp0("***********************")
            printp0("Timestep {}".format(t))

            u1, p1 = incns.step()
            T1 = eneq.step(u1)

            Tp1 = T1.at(0.181, 7.37, tolerance=1e-8)
            up1 = u1.at(0.181, 7.37, tolerance=1e-8)[0]
            vp1 = u1.at(0.181, 7.37, tolerance=1e-8)[1]
            Nu0 = assemble(Constant(1.0 / H) * nusselt * ds(1))

            if write_csv:
                csv_file.write('{:+.5E} {:+.5E} {:+.5E} {:+.5E} {:+.5E}\n'.format(t, Tp1, up1, vp1, Nu0))

            if step % output_frequency == 0:
                outfile.write(u1, p1, T1, time=t)
                if write_csv:
                    csv_file.flush()

            step += 1

        printp0("")


if __name__ == "__main__":
    # test_incns_eneq_setup()

    # test_incns_eneq_bouyancy()

    test_temperature_driven_cavity()
