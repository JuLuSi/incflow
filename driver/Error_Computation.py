import numpy
from firedrake.petsc import *


class Error_Comp():
    def __init__(self):
        self.Mat_Err = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.Mat_Err.setType('dense')
        self.FE_Sol = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.FE_Sol.setType('dense')

    def clear(self):
        self.Mat_Err.destroy()

    def set_error_matrix_size(self, Nx, Nt):
        # We need to call this before starting to store snapshots
        self.Mat_Err.setSizes([Nx, Nt])
        self.Mat_Err.setUp()

    def set_FE_Sol_matrix_size(self, Nx, Nt):
        # We need to call this before starting to store snapshots
        self.FE_Sol.setSizes([Nx, Nt])
        self.FE_Sol.setUp()

    def take_error_element(self, v, l):
        row, col = self.Mat_Err.getSize()
        idxr = range(row);
        self.Mat_Err.setValues(idxr, l, v)
        self.Mat_Err.assemble()

    def take_FE_Sol(self,v,l):
        row, col = self.FE_Sol.getSize()
        idxr = range(row);
        self.FE_Sol.setValues(idxr, l, v)
        self.FE_Sol.assemble()

    def prepare_L2norm_list(self, A, IP):
        # A is the matrix we want to compute the norm
        V = (IP.matMult(A).transposeMatMult(A)).getDiagonal()
        Vec_Err = ((V.getValues(range(V.size))).tolist())

        return Vec_Err

    def compute_L2timespace_norm(self, A, IP, dt, Nt):
        # A is the quantity we want to compute the L2timespace norm
        # IP is the inner product matrix
        # dt is the discretization time step
        # Nt is the number of time steps

        Vec_Err = self.prepare_L2norm_list(A, IP)
        err = 0.0
        for i in range(Nt - 1):
            err += (Vec_Err[i] + Vec_Err[i + 1]) * 0.5 * dt
        err = numpy.sqrt(err)

        return err

    def compute_L2averagetime_norm(self, A, IP, Nt):
        # A is the quantity we want to compute the L2averagetime norm
        # IP is the inner product matrix
        # Nt is the number of time steps

        Vec_Err = self.prepare_L2norm_list(A, IP)
        err = 0.0
        for i in range(Nt):
            err += numpy.sqrt(Vec_Err[i])
        err = err / Nt

        return err

    def compute_L2timespace_norm_relative(self, A, B, IP, dt, Nt):
        # A represent abs(vFE-vPOD)
        # B represent vFE

        err = self.compute_L2timespace_norm(A, IP, dt, Nt)
        err2 = self.compute_L2timespace_norm(B, IP, dt, Nt)
        errrel = err / err2

        return errrel

    def compute_L2averagetime_norm_relative(self, A, B, IP, Nt):
        # A represent for example abs(vFE-vPOD)
        # B represent for example vFE

        err = self.compute_L2averagetime_norm(A, IP, Nt)
        err2 = self.compute_L2averagetime_norm(B, IP, Nt)
        errrel = err / err2

        return errrel
