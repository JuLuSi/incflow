from __future__ import absolute_import, print_function, division
from six.moves import map, range
from firedrake.petsc import PETSc
import scipy.sparse as sp


def printp0(message):
    comm = PETSc.COMM_WORLD
    if comm.Get_rank() == 0:
        print(message)


def fdmat_to_petsc(A):
    return A.M.handle


def petscmat_to_sp(A):
    Asp = A.getValuesCSR()[::-1]
    return sp.csr_matrix(Asp)
