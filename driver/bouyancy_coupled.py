from os.path import abspath, basename, dirname, join
from firedrake import *
import ufl
from incflow import *

cwd = abspath(dirname(__file__))
data_dir = join(cwd, "../tests/data")

mesh = Mesh(data_dir + "/bouyancy.e")
mesh = MeshHierarchy(mesh, 3)[-1]

model = IncNavierStokesEnEq(mesh, nu=1.343e-5, rho=1.1644, cp=1.005, k=0.0257)

model.dt = 0.125

W = model.get_mixed_fs()

bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), (1, 2, 3, 4, 5, 6))

# bcT = [DirichletBC(S, Constant(293.5), (1, 2, 4)),
#        DirichletBC(S, Constant(297.0), (3,))]

bcT = DirichletBC(W.sub(2), Constant(297.0), (5,))

model.set_bcs(u_bcs=[bcu_walls],
              p_bcs=[],
              T_bcs=[bcT])

model.setup_solver()

outfile = File(join("results/", "test_incns_eneq.pvd"))

model.upT0.sub(2).assign(Constant(293.5))
bcT.apply(model.upT0)

step = 0
t = 0.0
t_end = 100.0 - 1e-08
num_timesteps = int(t_end / model.dt)
output_frequency = 50

print("Number of timesteps: {}".format(num_timesteps))
print("Output frequency: {}".format(output_frequency))
print("Number of output files: {}".format(int(num_timesteps / output_frequency)))
print("DOFs: {}".format(model.upT0.vector().size()))

while t <= t_end:
    t += model.dt

    printp0("***********************")
    printp0("Timestep {}".format(t))

    model.set_forcing(
        Constant(1.0 / 293.5) * (model.T0 - Constant(293.5)) * as_vector([Constant(0.0), Constant(-9.81)]))

    u1, p1, T1 = model.step()

    printp0("")

    if step % output_frequency == 0:
        outfile.write(u1, p1, T1, time=t)

    step += 1
