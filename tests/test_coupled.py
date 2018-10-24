from os.path import abspath, basename, dirname, join
from firedrake import *
from incflow import *
import numpy as np
import pytest

set_log_level(DEBUG)

cwd = abspath(dirname(__file__))
data_dir = join(cwd, "data")

H = 8.0
L = 1.0
Pr = 0.71
Ra = 3.4e5
mu = Pr
k = 1.0
rho = 1.0
cp = 1.0
g = 1.0
T_cold = Constant(-0.5)
T_hot = Constant(+0.5)

N = 100
mesh = RectangleMesh(N / 4, N, L, H)
x, y = SpatialCoordinate(mesh)

DG0 = FunctionSpace(mesh, "DG", 0)

write_csv = True

model = IncNavierStokesEnEq(mesh, nu=sqrt(Pr / Ra), rho=rho, k=1.0 / sqrt(Ra * Pr), cp=cp)
model.dt.assign(0.005)

W = model.get_mixed_fs()

bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), (1, 2, 3, 4))

bcT = [
    DirichletBC(W.sub(2), T_cold, (2,)),
    DirichletBC(W.sub(2), T_hot, (1,))
]

p_pin = DirichletBC(W.sub(1), Constant(0), (1,))
p_pin.nodes = p_pin.nodes[:1]

model.set_bcs(u_bcs=[bcu_walls],
              p_bcs=[],
              T_bcs=bcT)

model.has_nullspace = True

model.setup_solver(stabilization=False)

outfile = File(join(data_dir, "../", "results/", "test_temp_driven_cavity.pvd"))

model.upT0.sub(2).project(Constant(0.0))

step = 0
t = 0.0
t_end = 100.0
num_timesteps = int(t_end / float(model.dt))
output_frequency = 100

print("Number of timesteps: {}".format(num_timesteps))
print("Output frequency: {}".format(output_frequency))
print("Number of output files: {}".format(int(num_timesteps / output_frequency)))
print("DOFs: {}".format(model.upT0.vector().size()))

if write_csv:
    csv_file = open('mit81_points.csv', 'w', newline='')
    csv_file.write('time Tp1 up1 vp1 Nu0\n')

n = FacetNormal(mesh)
nusselt = dot(grad(model.upT0.sub(2)), n)

model.build_forcing_projector(model.T1 * as_vector([Constant(0.0), Constant(1.0)]))

while t <= t_end:
    with Timer('Step'):
        t += float(model.dt)

        printp0("***********************")
        printp0("Time {:.3E}".format(t))
        printp0("Timestep {:.3E}".format(float(model.dt)))
        
        model.set_forcing()

        model.solve()

        u1, p1, T1 = model.step()

        Tp1 = T1.at(0.181, 7.37, tolerance=1e-8)
        up1 = u1.at(0.181, 7.37, tolerance=1e-8)[0]
        vp1 = u1.at(0.181, 7.37, tolerance=1e-8)[1]
        Nu0 = assemble(Constant(1.0 / H) * nusselt * ds(1))

        # cfl = project(sqrt(inner(u1, u1)) * model.dt / CellSize(mesh), DG0)
        # max_cfl = np.max(cfl.vector().array())
           
        # print('CFL: {:+.5E}'.format(max_cfl))

        # cfl_max = 1.0
        # delta_dt = 0.01
        # if max_cfl >= cfl_max or float(model.dt) > 0.1:
        #     model.dt.assign(float(model.dt) - delta_dt)
        # else:
        #     model.dt.assign(float(model.dt) + delta_dt)

        if write_csv:
            csv_file.write('{:+.5E} {:+.5E} {:+.5E} {:+.5E} {:+.5E}\n'.format(t, Tp1, up1, vp1, Nu0))

        if step % output_frequency == 0:
            outfile.write(u1, p1, T1, time=t)
            if write_csv:
                csv_file.flush()

        step += 1

    printp0("")
