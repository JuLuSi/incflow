from __future__ import absolute_import, print_function, division
from os.path import abspath, basename, dirname, join
from incflow import *
from firedrake import *
from firedrake.petsc import PETSc
import scipy.interpolate as spit
import numpy as np
from energy_eq_adjoint import *

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

incns.set_bcs(u_bcs=[bcu_inflow, bcu_walls, bcu_cylinder],
              p_bcs=[bcp_outflow])

incns.setup_solver()

step = 0
t = 0.0
t_end = 0.025
Nt = int(t_end / incns.dt)

u0 = PETSc.Vec().create(PETSc.COMM_WORLD)
u0.setSizes(Nt)
u0.setUp()

u0.setValues(range(Nt), np.full(Nt,300))
#u0.setValues(50,20)

control = spit.BSpline(np.linspace(-incns.dt, t_end + incns.dt, Nt + 2), u0.getValues(range(Nt)), 1)

BcIn= Function(eneq.S)

BcIn.assign(float(control(0)))


bcT = [DirichletBC(S, BcIn, (1,)),
       DirichletBC(S, Constant(250.0), (2, 4))]

eneq.set_bcs(bcT)

eneq.setup_solver(W.sub(0))

outfile = File(join(data_dir, "../", "results/", "test_incns_eneq.pvd"))
vel1, p1 = incns.up.split()
outfile.write(vel1, p1, eneq.T1)

vel_states = []
T_states=[]
T_states.append(eneq.T1.copy(deepcopy=True))
vel_states.append(vel1.copy(deepcopy=True))
for i in range(1, Nt+1):
    t += incns.dt

    printp0("***********************")
    printp0("Timestep {}".format(t))

    vel1, p1 = incns.step()
    vel_states.append(vel1.copy(deepcopy=True))
    BcIn.assign(float(control(t)))
    T1 = eneq.step(vel1)
    T_states.append(T1.copy(deepcopy=True))

    printp0("")

    # if step % 5 == 0:
    outfile.write(vel1, p1, T1)

    step += 1

eneqAdj=  EnergyEqAdj(mesh)

bcq= [DirichletBC(S, Constant(0.0), (1, 2, 4))]

eneqAdj.set_bcs(bcq)

eneqAdj.set_T(T_states[Nt])

Td = Function(eneqAdj.S).interpolate(Constant(0.0))
eneqAdj.set_Td(Td)

eneqAdj.setup_solver(W.sub(0))

outfileAdj = File(join(data_dir, "../", "results/", "Adjoint.pvd"))
outfileAdj.write(eneqAdj.q1)

for i in range(Nt-1,-1,-1):
    t-=incns.dt

    printp0("***********************")
    printp0("Timestep {}".format(t))

    q0 = eneqAdj.step(vel_states[i],T_states[i])
    outfileAdj.write(q0)
