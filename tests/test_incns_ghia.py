from __future__ import absolute_import, print_function, division
from six.moves import map, range
from os.path import abspath, basename, dirname, join
from firedrake import *
from firedrake.petsc import PETSc
from incflow import *
import pytest
import numpy as np
import matplotlib.pyplot as plt

cwd = abspath(dirname(__file__))
data_dir = join(cwd, "data")


def plot_results(u1, re, N):
    mr = {100: 1, 1000: 2, 3200: 3, 5000: 4, 10000: 5}

    tmp_x = []
    tmp_y = []

    M = np.genfromtxt(join(data_dir, "ghia.txt"), comments='#',
                      usecols=(6, mr[re], 0, 6 + mr[re])).transpose()

    # for i in xrange(len(M[:][0])):
    grid = np.linspace(0, 1, N)
    for i in range(len(grid)):
        tmp_x.append(u1.at([grid[i], 0.5])[1])
        tmp_y.append(u1.at([0.5, grid[i]])[0])

    plt.subplot(2, 1, 1)
    plt.plot(grid, tmp_x, M[0][:], M[3][:], 'ro')
    plt.title('y = 0.5')
    plt.xlabel('x/L')
    plt.ylabel('Uy')

    plt.subplot(2, 1, 2)
    plt.plot(grid, tmp_y, M[2][:], M[1][:], 'ro')
    # plt.title('x = 0.5')
    plt.xlabel('y/L')
    plt.ylabel('Ux')

    plt.savefig(join(data_dir, "../", "results/", "ghia.png"))


def test_incns_ghia_benchmark():
    N = 50
    Re = 10000

    mesh = UnitSquareMesh(N, N)

    incns = IncNavierStokes(mesh)

    incns.nu = 1.0 / Re
    incns.rho = 1.0
    incns.mu = incns.nu * incns.rho
    incns.dt = 0.01
    incns.time_integration_method = "backward_euler"
    incns.pressure_nullspace = True
    incns.supg = True

    V = incns.get_u_fs()
    Q = incns.get_p_fs()

    bcu_top = DirichletBC(V, Constant((1.0, 0.0)), (4,))
    bcu_borders = DirichletBC(V, Constant(0.0), (1, 2, 3))
    bcs_p = []
    bcs_u = [bcu_top, bcu_borders]

    incns.set_bcs(bcs_u, bcs_p)

    incns.setup_solver()

    outfile = File(join(data_dir, "../", "results/",
                        "test_incns_ghia_benchmark.pvd"))

    step = 0
    t = 0.0
    t_end = 1.0

    while(t < t_end - incns.dt):
        t += incns.dt

        printp0("***********************")
        printp0("Timestep {}".format(t))

        u1, p1 = incns.step()

        if step % 100 == 0:
            outfile.write(u1, p1)

        step += 1

    plot_results(u1, Re, N)


if __name__ == "__main__":
    test_incns_ghia_benchmark()
