import numpy
from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc


class POD():
    def __init__(self):
        self.snapshot_mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.snapshot_mat.setType('dense')
        self.full_basis_mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.full_basis_mat.setType('dense')
        self.basis_mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        self.basis_mat.setType('dense')
        self.L = 0

    def clear(self):
        self.snapshot_mat.destroy()
        self.full_basis_mat.destroy()
        self.basis_mat.destroy()
        self.L = 0

    def set_snapshot_matrix_size(self, Nx, Nt):
        # We need to call this before starting to store snapshots
        self.snapshot_mat.setSizes([Nx, Nt])
        self.snapshot_mat.setUp()

    def store_one_snapshot(self, snap, l):
        # l is the index of the l-th column where we want to place the snapshot
        self.snapshot_mat.setValues(
            range(self.snapshot_mat.size[0]), l, snap.vector().array())
        self.snapshot_mat.assemble()

    def compute(self, IP, D, eigenvalues_file, N, tol):
        """Compute projection matrix.

        Args:
            IP: Inner product matrix.
            D: Trapeizoidal weights matrix.
            eigenvalues_file: Filename for eigenvalues output.
            N: Number of POD Basis we want (if possible).
            tol: Tolerance for the negligible part of complex value in the eigenvalues.
        """

        Cp = D.matMult(self.snapshot_mat.transposeMatMult(IP.matMult(self.snapshot_mat.matMult(D))))

        E = SLEPc.EPS()
        E.create()

        E.setOperators(Cp)
        E.setDimensions(Cp.size[0], -1, -1)
        # E.setProblemType(SLEPc.EPS.ProblemType.HEP)
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

        scaling_factors = PETSc.Vec().create(PETSc.COMM_WORLD)
        scaling_factors.setSizes(Lmax)
        scaling_factors.setUp()
        self.full_basis_mat.setSizes([self.snapshot_mat.size[0], Lmax])
        self.full_basis_mat.setUp()

        self.full_basis_mat = self.snapshot_mat.matMult(D.matMult(eigv))
        for i in range(Lmax):
            print("lambda_", i, " = ", eigs.getValue(i))
            scaling_factors.setValue(i, 1 / sqrt(eigs.getValue(i)))
        self.full_basis_mat.diagonalScale(None, scaling_factors)

        self.basis_mat.setSizes([self.snapshot_mat.size[0], self.L])
        self.basis_mat.setUp()

        self.basis_mat.setValues(range(self.snapshot_mat.size[0]), range(self.L),
                                 self.full_basis_mat.getValues(range(self.snapshot_mat.size[0]),
                                                               range(self.L)))
        self.basis_mat.assemble()

        print("Choosen ", self.L, "basis of ", Lmax, " with ratio=", ratio)
