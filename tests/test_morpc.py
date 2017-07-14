from __future__ import absolute_import, print_function, division
from firedrake import *
from firedrake.petsc import PETSc
from incflow import *
import pytest

N = 20
mesh = UnitSquareMesh(N, N)

V = VectorFunctionSpace(mesh, 'CG', 2)
u1 = Function(V).interpolate(as_vector([0.0, 0.0]))

eneq = EnergyEq(mesh)

eneq.dt = 0.05

S = eneq.get_fs()

bcT = [DirichletBC(S, Constant(300.0), (1,)),
       DirichletBC(S, Constant(0.0), (2, 3, 4))]

eneq.set_bcs(bcT)
eneq.setup_solver(V)

eneq.set_u(u1)

step = 0
t = 0.0
t_end = 0.5
num_timesteps = int(t_end / eneq.dt)

# Create pseudo operator
rows = eneq.T0.vector().size()
Z = PETSc.Mat().create(PETSc.COMM_WORLD)
Z.setType("aij")
Z.setSizes([rows, rows])
Z.setUp()
Z.assemble()
vd = Z.getDiagonal()
vd.set(1.0)
Z.setDiagonal(vd)

appctx = {"projection_mat": Z}

eneq_rom_solver = NonlinearVariationalSolver(
    eneq.energy_eq_problem,
    appctx=appctx,
    solver_parameters={
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "incflow.MORPC"
    }
)

outfile = File("basis.pvd")
outfile.write(eneq.T1, u1)

for k in range(num_timesteps):
    t += eneq.dt

    eneq_rom_solver.solve()

    outfile.write(eneq.T1, u1)

    eneq.T0.assign(eneq.T1)

    step += 1