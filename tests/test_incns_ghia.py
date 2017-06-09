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


def plot_results(u1, re):
    mr = {100: 1, 1000: 2, 3200: 3, 5000: 4, 10000: 5}

    tmp_x = []
    tmp_y = []

    M = np.genfromtxt(join(data_dir, "ghia.txt"), comments='#', usecols=(6, mr[re], 0, 6 + mr[re])).transpose()

    for i in xrange(len(M[:][0])):
        tmp_x.append(u1.at([M[0][i], 0.5])[1])
        tmp_y.append(u1.at([0.5, M[2][i]])[0])

    plt.subplot(2, 1, 1)
    plt.plot(M[0][:], tmp_x, M[0][:], M[3][:])
    plt.title('y = 0.5')
    plt.xlabel('x/L')
    plt.ylabel('Uy')

    plt.subplot(2, 1, 2)
    plt.plot(M[2][:], tmp_y, M[2][:], M[1][:])
    # plt.title('x = 0.5')
    plt.xlabel('y/L')
    plt.ylabel('Ux')

    plt.show()


def test_incns_ghia_benchmark():
    N = 50
    Re = 1000

    mesh = UnitSquareMesh(N, N)

    incns = IncNavierStokes(mesh)

    incns.nu = 1.0 / Re
    incns.rho = 1.0
    incns.mu = incns.nu * incns.rho
    incns.dt = 0.1

    incns.has_nullspace = True

    W = incns.get_mixed_fs()

    bcu_top = DirichletBC(W.sub(0), Constant((1.0, 0.0)), (4,))
    bcu_borders = DirichletBC(W.sub(0), Constant(0.0), (1, 2, 3))
    bcs_p = []
    bcs_u = [bcu_top, bcu_borders]

    incns.set_bcs(bcs_u, bcs_p)

    incns.setup_solver()

    outfile = File(join(data_dir, "../", "results/", "test_incns_ghia_benchmark.pvd"))

    step = 0
    t = 0.0
    t_end = 35.0

    while (t < t_end):
        t += incns.dt

        printp0("***********************")
        printp0("Timestep {}".format(t))

        u1, p1 = incns.step()

        if step % 1 == 0:
            outfile.write(u1, p1)

        step += 1

    plot_results(u1, Re)


if __name__ == "__main__":
    test_incns_ghia_benchmark()
