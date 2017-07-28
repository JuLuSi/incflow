from slepc4py import SLEPc
from firedrake.petsc import PETSc
import numpy
from firedrake import *


class POD_SLEPc():
    def __init__(self):
        self.snapshots_matrix = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.Basis_matrix = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.L = 0
        # self.snapshots_matrix.setType('dense')

    def clear(self):
        self.snapshots_matrix.destroy()
        self.Basis_matrix.destroy()
        self.L = 0

    def set_snapshot_matrix_size(self, Nx, Nt):
        # We need to call this before starting to store snapshots
        self.snapshots_matrix.setSizes([Nx, Nt])
        self.snapshots_matrix.setUp()

    def store_one_snapshot(self, snap, l):
        # l is the index of the l-th column where we want to place the snapshot
        row, col = self.snapshots_matrix.getSize()
        idxr = range(row);
        self.snapshots_matrix.setValues(idxr, l, snap.vector().array())
        if l == col - 1:
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
        E.setFromOptions()

        E.solve()

        Lmax = E.getConverged()
        print "Number of converged eigenpairs ", Lmax

        eigvr, _ = Cp.getVecs()
        eigvimg, _ = Cp.getVecs()

        eigs = PETSc.Vec().create(PETSc.COMM_WORLD)
        eigs.setSizes(Lmax)
        eigs.setUp()
        eigv = PETSc.Mat().create(PETSc.COMM_WORLD)
        eigv.setSizes([eigvr.size, Lmax])
        eigv.setUp()

        count = 0
        for i in range(Lmax):
            a = E.getEigenpair(i, eigvr, eigvimg)
            vr = a.real
            vi = a.imag
            if (vr > 0 and vi <= tol):
                # In SLEPc eigenvalues are sorted by magnitude, we want only the positive and
                # we check if there are no negligible complex parts
                eigs.setValue(i, vr)
                eigv.setValues(range(eigv.size[0]), i, eigvr.getValues(range(eigv.size[0])))
                count = count + 1

        Lmax = min(Lmax, count)
        self.L = min(Lmax, N)

        eigv.assemble()

        numpy.savetxt(eigenvalues_file, eigs.getValues(range(Lmax)))

        ratio = eigs.getValues(range(self.L)).sum() / eigs.sum()

        psi = PETSc.Vec().create(PETSc.COMM_WORLD)
        psi.setSizes(self.L)
        psi.setUp()
        appo = PETSc.Vec().create(PETSc.COMM_WORLD)
        appo.setSizes(self.snapshots_matrix.size[0])
        appo.setUp()
        self.Basis_matrix.setSizes([self.snapshots_matrix.size[0], self.L])
        self.Basis_matrix.setUp()

        V = self.snapshots_matrix.matMult(D)
        for i in range(self.L):
            print "lambda_", i, " = ", eigs.getValue(i)
            D.mult(eigv.getColumnVector(i), psi)
            psi.scale(1 / sqrt(eigs.getValue(i)))
            V.mult(psi, appo)
            self.Basis_matrix.setValues(range(self.Basis_matrix.size[0]), i,
                                        appo.getValues(range(self.Basis_matrix.size[0])))

        self.Basis_matrix.assemble()

        print "Choosen ", self.L, "basis of ", Lmax, " with ratio=", ratio
