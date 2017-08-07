from __future__ import absolute_import, print_function, division
from os.path import abspath, basename, dirname, join
from incflow import *
from firedrake import *
from firedrake.petsc import PETSc
from error_computation import *

cwd = abspath(dirname(__file__))
data_dir = join(cwd, "data")

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

incns = IncNavierStokes(mesh, nu=0.0001, rho=1.0)
eneq = EnergyEq(mesh)

incns.dt = 0.005
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
u1, p1 = incns.up.split()
outfile.write(u1, p1, eneq.T1)

step = 0
t = 0.0
t_end = 5.00

podobj = POD()
M = eneq.get_mass_matrix()
IPL2 = fdmat_to_petsc(assemble(M))
M = eneq.get_H1_matrix()
IP = fdmat_to_petsc(assemble(M))

Nt = int(t_end / incns.dt)
POD_ts= int(Nt/2.)

podobj.set_snapshot_matrix_size(eneq.T0.vector().size(), POD_ts)
errobj = Error_Comp()
errobj.set_error_matrix_size(eneq.T0.vector().size(), Nt)
errobj.set_FE_Sol_matrix_size(eneq.T0.vector().size(), Nt)
vel_states = []
for i in range(0, Nt):
    t += incns.dt

    printp0("***********************")
    printp0("Timestep {}".format(t))

    u1, p1 = incns.step()
    vel_states.append(u1.copy(deepcopy=True))
    T1 = eneq.step(u1)

    if step<POD_ts:
        podobj.store_one_snapshot(T1, i)

    errobj.take_FE_Sol(T1.vector().array(),i)

    printp0("")

    if step % 5 == 0:
        outfile.write(u1, p1, T1)

    step += 1

D = PETSc.Mat().create(PETSc.COMM_WORLD)
D.setType('aij')
D.setSizes(POD_ts, POD_ts)
D.setUp()
D.setValues(0, 0, sqrt(incns.dt * 0.5))
for i in range(1, D.size[0] - 1):
    D.setValues(i, i, sqrt(incns.dt))
D.setValues(D.size[0] - 1, D.size[0] - 1, sqrt(incns.dt * 0.5))
D.assemble()

L = 50  # Number of POD Basis we want (if possible)
tol = 1.e-14  # Tolerance for the negligible part of complex value in the eigenvalues

podobj.compute(IP, D, "./results/eig.txt", L, tol)
snap = File(join(data_dir, "../", "results/", "basis.pvd"))

L = podobj.L
Z = podobj.basis_mat

basis = Function(S)

for i in range(0, L):
    basis.dat.data[:] = Z.getValues(range(Z.size[0]), i)[:]
    snap.write(basis)

appctx = {"projection_mat": Z}

eneq_rom_solver = NonlinearVariationalSolver(
    eneq.energy_eq_problem,
    appctx=appctx,
    solver_parameters={
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "incflow.MORPC"
    }
)

incns.up0.assign(0)
incns.up.assign(0)
eneq.T0.assign(0)
eneq.T1.assign(0)

outfile = File(join(data_dir, "../", "results/", "POD.pvd"))
outfile.write(eneq.T1, u1)

step = 0
t = 0.0

for k in range(Nt):
    t += eneq.dt

    printp0("***********************")
    printp0("Timestep {}".format(t))

    eneq.set_u(vel_states[k])
    eneq_rom_solver.solve()

    if step % 5 == 0:
        outfile.write(eneq.T1, vel_states[k])

    errobj.take_error_element(
        abs(errobj.FE_Sol.getValues(range(podobj.snapshot_mat.size[0]), k) - eneq.T1.vector().array()), k)

    eneq.T0.assign(eneq.T1)

    printp0("")

    step += 1

errrel = errobj.compute_L2timespace_norm_relative(errobj.Mat_Err, errobj.FE_Sol, IPL2, eneq.dt, Nt)
err_avgrel = errobj.compute_L2averagetime_norm_relative(errobj.Mat_Err, errobj.FE_Sol, IPL2, Nt)

printp0("Relative Error POD-FE in L2(0,T,Om) Norm {}".format(errrel))
printp0("Relative Averaged Error POD-FE in L2(Om) Norm {}".format(err_avgrel))
