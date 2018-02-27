from os.path import abspath, basename, dirname, join
from firedrake import *
from scipy.io import savemat
import numpy as np
from incflow import *

cwd = abspath(dirname(__file__))
data_dir = join(cwd, "data")

mesh = Mesh(data_dir + "/room.e")
mesh = MeshHierarchy(mesh, 1)[-1]

incns = IncNavierStokes(mesh, nu=1.81e-5, rho=1.1839)

incns.dt = 0.01

W = incns.get_mixed_fs()

inflow_profile = (
    "1.5 * 4.0 * x[1] * (0.5 - x[1]) / ( 0.5 * 0.5 )", "0.0")

bcu_inflow = DirichletBC(W.sub(0), Expression(inflow_profile, degree=2), (1,))
bcp_outflow = DirichletBC(W.sub(1), Constant(0.0), (2,))
bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), (3,))

incns.set_bcs(u_bcs=[bcu_inflow, bcu_walls],
              p_bcs=[bcp_outflow])

incns.setup_solver()

outfile = File(join(data_dir, "../", "results/", "test_konstanz.pvd"))

step = 0
t = 0.0
t_end = 0.5
num_timesteps = int(t_end / incns.dt)
output_frequency = 5

print("Number of timesteps: {}".format(num_timesteps))
print("Output frequency: {}".format(output_frequency))
print("Number of output files: {}".format(int(num_timesteps / output_frequency)))
print("INS DOFs: {}".format(incns.up0.vector().size()))

snapshots = []

while t <= t_end:
    t += incns.dt

    printp0("***********************")
    printp0("Timestep {}".format(t))

    u1, p1 = incns.step()

    printp0("")

    if step % output_frequency == 0:
        outfile.write(u1, p1)

    snapshots.append(u1.copy(deepcopy=True))

    step += 1

snapshots_array = []

for snapshot in snapshots:
    with snapshot.dat.vec as snapshot_loc:
        snapshots_array.append(snapshot_loc.array)

savemat(join(data_dir, '../', 'results/', 'snapshots.mat'),
        {'S': np.array(snapshots_array)})
