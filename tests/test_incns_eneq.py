from __future__ import absolute_import, print_function, division
from os.path import abspath, basename, dirname, join
from firedrake import *
from incflow import *
import pytest

cwd = abspath(dirname(__file__))
data_dir = join(cwd, "data")


@pytest.mark.regression
def test_incns_eneq_setup():
    mesh = Mesh(data_dir + "/cyl.e")

    dm = mesh._plex
    from firedrake.mg.impl import filter_exterior_facet_labels
    for _ in range(2):
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

    incns = IncNavierStokes(mesh, nu=0.00005, rho=1.0)
    eneq = EnergyEq(mesh)

    incns.dt = 0.0005
    eneq.dt = incns.dt

    W = incns.get_mixed_fs()
    S = eneq.get_fs()

    inflow_profile = (
        "1.5 * 4.0 * x[1] * (0.41 - x[1]) / ( 0.41 * 0.41 )", "0.0")
    bcu_inflow = DirichletBC(W.sub(0), Expression(inflow_profile, degree=2), (1,))
    bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), (2, 4))
    bcu_cylinder = DirichletBC(W.sub(0), Constant((0, 0)), (5,))
    bcp_outflow = DirichletBC(W.sub(1), Constant(0), (3,))

    bcT = [DirichletBC(S, Constant(300.0), (1,)),
           DirichletBC(S, Constant(250.0), (2, 4))]

    incns.set_bcs(u_bcs=[bcu_inflow, bcu_walls, bcu_cylinder],
                  p_bcs=[bcp_outflow])

    eneq.set_bcs(bcT)

    incns.setup_solver()
    eneq.setup_solver(W.sub(0))
    outfile = File(join(data_dir, "../", "results/", "test_incns_eneq.pvd"))

    step = 0
    t = 0.0
    t_end = 35.0
    num_timesteps = int(t_end / incns.dt)
    output_frequency = 50

    print("Number of timesteps: {}".format(num_timesteps))
    print("Output frequency: {}".format(output_frequency))
    print("Number of output files: {}".format(int(num_timesteps / output_frequency)))
    print("INS DOFs: {}".format(incns.up0.vector().size()))
    print("ENEq DOFs: {}".format(eneq.T0.vector().size()))

    while(t <= t_end):
        t += incns.dt

        printp0("***********************")
        printp0("Timestep {}".format(t))

        u1, p1 = incns.step()
        T1 = eneq.step(u1)

        printp0("")

        if step % output_frequency == 0:
            outfile.write(u1, p1, T1)

        step += 1


if __name__ == "__main__":
    test_incns_eneq_setup()
