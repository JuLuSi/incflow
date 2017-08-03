import numpy
from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc


class POD():
    def __init__(self):
        self.snapshots_mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.full_basis_mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.basis_mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.L = 0
        # self.snapshots_mat.setType('dense')

    def clear(self):
        self.snapshots_mat.destroy()
        self.full_basis_mat.destroy()
        self.basis_mat.destroy()
        self.L = 0

    def set_snapshot_matrix_size(self, Nx, Nt):
        # We need to call this before starting to store snapshots
        self.snapshots_mat.setSizes([Nx, Nt])
        self.snapshots_mat.setUp()

    def store_one_snapshot(self, snap, l):
        # l is the index of the l-th column where we want to place the snapshot
        self.snapshots_mat.setValues(
            range(self.snapshots_mat.size[0]), l, snap.vector().array())
        self.snapshots_mat.assemble()

    def compute(self, IP, D, eigenvalues_file, N, tol):
        """Compute projection matrix.

        Args:
            IP: Inner product matrix.
            D: Trapeizoidal weights matrix.
            eigenvalues_file: Filename for eigenvalues output.
            N: Number of POD Basis we want (if possible).
            tol: Tolerance for the negligible part of complex value in the eigenvalues.
        """

        Cp = D.matMult(IP.matMult(self.snapshots_mat.matMult(
            D)).transposeMatMult(self.snapshots_mat))

        E = SLEPc.EPS()
        E.create()

        E.setOperators(Cp)
        E.setDimensions(Cp.size[0], -1, -1)
        E.setTolerances(1.e-15, None)
        E.setProblemType(2)  # Generalized Hermitian Problem
        E.setFromOptions()

        E.solve()

        nconv = E.getConverged()
        print("Number of converged eigenpairs ", nconv)

        eigvr, _ = Cp.getVecs()
        eigvimg, _ = Cp.getVecs()

        count = nconv
        for i in range(nconv):
            a = E.getEigenpair(i, eigvr, eigvimg)
            vr = a.real
            vi = a.imag
            if vr <= 0 or abs(vi) > tol:
                # We want only the positive and we check if there are no negligible complex parts
                count = min(count, i)
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
            eigv.setValues(range(eigv.size[0]), i,
                           eigvr.getValues(range(eigv.size[0])))

        eigv.assemble()

        numpy.savetxt(eigenvalues_file, eigs.getValues(range(Lmax)))

        ratio = eigs.getValues(range(self.L)).sum() / eigs.sum()

        appo = PETSc.Vec().create(PETSc.COMM_WORLD)
        appo.setSizes(Lmax)
        appo.setUp()
        self.full_basis_mat.setSizes([self.snapshots_mat.size[0], Lmax])
        self.full_basis_mat.setUp()

        self.full_basis_mat = self.snapshots_mat.matMult(D.matMult(eigv))
        for i in range(Lmax):
            print("lambda_", i, " = ", eigs.getValue(i))
            appo.setValue(i, 1 / sqrt(eigs.getValue(i)))
        self.full_basis_mat.diagonalScale(None, appo)

        self.basis_mat.setSizes([self.snapshots_mat.size[0], self.L])
        self.basis_mat.setUp()

        self.basis_mat.setValues(range(self.snapshots_mat.size[0]), range(self.L),
                                    self.full_basis_mat.getValues(range(self.snapshots_mat.size[0]),
                                                                     range(self.L)))
        self.basis_mat.assemble()

        print("Choosen ", self.L, "basis of ", Lmax, " with ratio=", ratio)
