from __future__ import absolute_import, print_function, division
from os.path import abspath, basename, dirname, join
from firedrake import *
from firedrake.petsc import PETSc
from incflow import *

cwd = abspath(dirname(__file__))
data_dir = join(cwd, "data")


def test_incns_eneq_setup():
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
    eneq = EnergyEq(mesh)

    V = incns.get_u_fs()
    Q = incns.get_p_fs()
    S = eneq.get_T_fs()

    inflow_profile = (
        "1.5 * 4.0 * x[1] * (0.41 - x[1]) / ( 0.41 * 0.41 )", "0.0")
    bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), (1,))
    bcu_walls = DirichletBC(V, Constant((0.0, 0.0)), (2, 4))
    bcu_cylinder = DirichletBC(V, Constant((0, 0)), (5,))
    bcp_outflow = DirichletBC(Q, Constant(0), (3,))

    bcT = [DirichletBC(S, Constant(300.0), (1,)),
           DirichletBC(S, Constant(250.0), (2, 4))]

    incns.set_bcs(u_bcs=[bcu_inflow, bcu_walls, bcu_cylinder],
                  p_bcs=[bcp_outflow])

    eneq.set_bcs(bcT)

    incns.setup_solver()
    eneq.setup_solver(V)
    outfile = File(join(data_dir, "../", "results/", "test_incns_eneq.pvd"))

    step = 0
    t = 0.0
    t_end = 35.0

    while(t <= t_end):
        t += incns.dt

        printp0("***********************")
        printp0("Timestep {}".format(t))

        u1, p1 = incns.step()
        T1 = eneq.step(u1)

        printp0("")

        if step % 5 == 0:
            outfile.write(u1, p1, T1)

        step += 1


if __name__ == "__main__":
    test_incns_eneq_setup()
