from slepc4py import SLEPc
from firedrake.petsc import PETSc
import numpy
from firedrake import *


class POD_SLEPc():
    def __init__(self):
        self.snapshots_matrix = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.Full_Basis_matrix = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.Basis_matrix = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.L = 0
        # self.snapshots_matrix.setType('dense')

    def clear(self):
        self.snapshots_matrix.destroy()
        self.Full_Basis_matrix.destroy()
        self.Basis_matrix.destroy()
        self.L = 0

    def set_snapshot_matrix_size(self, Nx, Nt):
        # We need to call this before starting to store snapshots
        self.snapshots_matrix.setSizes([Nx, Nt])
        self.snapshots_matrix.setUp()

    def store_one_snapshot(self, snap, l):
        # l is the index of the l-th column where we want to place the snapshot
        self.snapshots_matrix.setValues(range(self.snapshots_matrix.size[0]), l, snap.vector().array())
        self.snapshots_matrix.assemble()

    def compute(self, IP, D, eigenvalues_file, N, tol):
        # IP= Inner product matrix
        # D= Trapeizoidal weights matrix
        # N number of POD Basis we want (if possible)
        # Tolerance for the negligible part of complex value in the eigenvalues


        Cp = D.matMult(IP.matMult(self.snapshots_matrix.matMult(D)).transposeMatMult(self.snapshots_matrix))

        E = SLEPc.EPS();
        E.create()

        E.setOperators(Cp)
        E.setDimensions(Cp.size[0], -1, -1)
        E.setTolerances(1.e-15,None)
        E.setProblemType(2) # Generalized Hermitian Problem
        E.setFromOptions()

        E.solve()

        nconv = E.getConverged()
        print "Number of converged eigenpairs ", nconv

        eigvr, _ = Cp.getVecs()
        eigvimg, _ = Cp.getVecs()

        count = nconv
        for i in range(nconv):
            a = E.getEigenpair(i, eigvr, eigvimg)
            vr = a.real
            vi = a.imag
            if vr <= 0 or abs(vi) > tol:
                #We want only the positive and we check if there are no negligible complex parts
                count = min(count,i)
                break

        Lmax = min(nconv, count)
        self.L = min(Lmax, N)

        eigs = PETSc.Vec().create(PETSc.COMM_WORLD)
        eigs.setSizes(Lmax)
        eigs.setUp()
        eigv = PETSc.Mat().create(PETSc.COMM_WORLD)
        eigv.setSizes([eigvr.size, Lmax])
        eigv.setUp()

        for i in range(Lmax):
            a = E.getEigenpair(i, eigvr, eigvimg)
            vr = a.real
            vi = a.imag
            # Now we can store the values, once we know the dimension of eigs and eigv we can take
            eigs.setValue(i, vr)
            eigv.setValues(range(eigv.size[0]), i, eigvr.getValues(range(eigv.size[0])))

        eigv.assemble()

        numpy.savetxt(eigenvalues_file, eigs.getValues(range(Lmax)))

        ratio = eigs.getValues(range(self.L)).sum() / eigs.sum()

        appo = PETSc.Vec().create(PETSc.COMM_WORLD)
        appo.setSizes(Lmax)
        appo.setUp()
        self.Full_Basis_matrix.setSizes([self.snapshots_matrix.size[0], Lmax])
        self.Full_Basis_matrix.setUp()

        self.Full_Basis_matrix = self.snapshots_matrix.matMult(D.matMult(eigv))
        for i in range(Lmax):
            print "lambda_", i, " = ", eigs.getValue(i)
            appo.setValue(i, 1 / sqrt(eigs.getValue(i)))
        self.Full_Basis_matrix.diagonalScale(None, appo)

        self.Basis_matrix.setSizes([self.snapshots_matrix.size[0], self.L])
        self.Basis_matrix.setUp()

        self.Basis_matrix.setValues(range(self.snapshots_matrix.size[0]), range(self.L),
                                    self.Full_Basis_matrix.getValues(range(self.snapshots_matrix.size[0]),
                                                                     range(self.L)))
        self.Basis_matrix.assemble()

        print "Choosen ", self.L, "basis of ", Lmax, " with ratio=", ratio
