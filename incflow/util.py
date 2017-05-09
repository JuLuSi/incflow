from __future__ import absolute_import, print_function, division
from six.moves import map, range
from firedrake.petsc import PETSc


def printp0(message):
    comm = PETSc.COMM_WORLD
    if comm.Get_rank() == 0:
        print(message)
