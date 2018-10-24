from os.path import abspath, basename, dirname, join
from firedrake import *
from incflow import *
import numpy as np
import pytest

set_log_level(DEBUG)

cwd = abspath(dirname(__file__))
data_dir = join(cwd, "data")

k = 0.02537 # W/(m*K)
rho = 1.225 # kg/m^3
cp = 1005.0 # J/(kg*K)
nu = 1.789 / rho * 1e-5 # m^2/s
g = 9.81 # m/s^2
thermal_expansion = 3.47e-3 # 1/K
# thermal_expansion = 1.0 / 293.0
T_cold = Constant(293)
T_hot = Constant(294)

N = 30
mesh = RectangleMesh(N, N, 0.1, 0.1)
x, y = SpatialCoordinate(mesh)

DG0 = FunctionSpace(mesh, "DG", 0)

model = IncNavierStokesEnEq(mesh, nu=nu, rho=rho, k=k, cp=cp)
model.dt.assign(0.05)

W = model.get_mixed_fs()

bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), (1, 2, 3, 4))

bcT = [
    DirichletBC(W.sub(2), T_cold, (1,)),
    DirichletBC(W.sub(2), T_hot, (2,))
]

p_pin = DirichletBC(W.sub(1), Constant(0), (1,))
p_pin.nodes = p_pin.nodes[:1]

model.set_bcs(u_bcs=[bcu_walls],
              p_bcs=[],
              T_bcs=bcT)

model.has_nullspace = True

model.setup_solver(stabilization=False)

model.upT0.sub(2).project(T_cold)
[bc.apply(model.upT0) for bc in bcT]
model.upT1.sub(2).project(T_cold)
[bc.apply(model.upT1) for bc in bcT]

outfile = File(join(data_dir, "../", "results/", "test_heated_cavity.pvd"))

step = 0
t = 0.0
t_end = 50.0
num_timesteps = int(t_end / float(model.dt))
output_frequency = 10

u1, p1, T1 = model.upTmr.split()
outfile.write(u1, p1, T1, time=t)

print("Number of timesteps: {}".format(num_timesteps))
print("Output frequency: {}".format(output_frequency))
print("Number of output files: {}".format(int(num_timesteps / output_frequency)))
print("DOFs: {}".format(model.upT0.vector().size()))

model.build_forcing_projector(as_vector([Constant(0.0), Constant(g)]) * Constant(thermal_expansion) * (model.T1 - T_cold))

while t <= t_end:
    with Timer('Step'):
        t += float(model.dt)

        printp0("***********************")
        printp0("Time {:.3E}".format(t))
        printp0("Timestep {:.3E}".format(float(model.dt)))
        
        model.set_forcing()

        model.solve()

        u1, p1, T1 = model.step()

        cfl = project(sqrt(inner(u1, u1)) * model.dt / CellVolume(mesh), DG0)
        max_cfl = np.max(cfl.vector().array())
           
        print('CFL: {:+.5E}'.format(max_cfl))

        if step % output_frequency == 0:
            outfile.write(u1, p1, T1, time=t)
            chk.store(u1)

        step += 1

    printp0("")
