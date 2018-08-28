from six.moves import map, range
from firedrake import assemble
from firedrake.petsc import PETSc
import scipy.sparse as sp
import time


def printp0(message):
    comm = PETSc.COMM_WORLD
    if comm.Get_rank() == 0:
        print(message)


def fdmat_to_petsc(A):
    return A.M.handle


def petscmat_to_sp(A):
    Asp = A.getValuesCSR()[::-1]
    return sp.csr_matrix(Asp)


def form_to_sp(A):
    return petscmat_to_sp(fdmat_to_petsc(assemble(A)))


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name)
            print('Elapsed: %s' % (time.time() - self.tstart))
