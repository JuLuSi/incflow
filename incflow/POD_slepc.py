import numpy
from slepc4py import SLEPc
from firedrake.petsc import PETSc
from incflow import *


class POD_SLEPc():
    def __init__(self):
        self.snapshots_matrix = numpy.array([])

    def clear(self):
        self.snapshots_matrix = numpy.array([])

    def take_one_snapshot(self, snap):
        if self.snapshots_matrix.size == 0:
            self.snapshots_matrix = numpy.array(snap.vector().array()).reshape(-1,
                                                                               1)  # we store the snapshot as column of snapshot matrix
        else:
            self.snapshots_matrix = numpy.hstack(
                (self.snapshots_matrix, numpy.array(snap.vector().array()).reshape(-1, 1)))

    def take_more_snapshots(self, snaps):
        if self.snapshots_matrix.size == 0:
            self.snapshots_matrix = numpy.array(snaps)
        else:
            self.snapshots_matrix = numpy.hstack(
                (self.snapshots_matrix, numpy.array(snaps)))  # we store the snapshots as columns of snapshots matrix

    def compute(self, IP, D, eigenvalues_file, N, tol):
        # IP= Inner product matrix
        # D= Trapeizoidal weights matrix
        # N number of POD Basis we want (if possible)
        # Tolerance for the negligible part of complex value in the eigenvalues

        C = D.dot(self.snapshots_matrix.T.dot(IP.dot(self.snapshots_matrix.dot(D))))
        rows = C.shape[0]
        cols = C.shape[1]
        Cp = PETSc.Mat().create(PETSc.COMM_WORLD)
        Cp.setType('dense')
        Cp.setSizes([rows, cols])
        Cp.setUp()

        idxr = range(rows)
        idxc = range(cols)

        Cp.setValues(idxr, idxc, C)
        Cp.assemble()

        E = SLEPc.EPS();
        E.create()

        E.setOperators(Cp)
        # E.setProblemType(SLEPc.EPS.ProblemType.HEP)
        E.setDimensions(C.shape[0], -1, -1)
        E.setFromOptions()

        E.solve()

        Lmax = E.getConverged()
        print "Number of converged eigenpairs ", Lmax

        eigvr, _ = Cp.getVecs()
        eigvimg, _ = Cp.getVecs()

        eigs = []
        eigv = numpy.array([])
        for i in range(Lmax):
            a = E.getEigenpair(i, eigvr, eigvimg)
            eigs.append(a)
            if eigv.size == 0:
                v = eigvr.getArray()
                eigv = v.reshape(-1, 1)
                eigvr.destroy()
                eigvr, _ = Cp.getVecs()
            else:
                v = eigvr.getArray()
                eigv = numpy.hstack((eigv, v.reshape(-1, 1)))

        eigs = numpy.asarray(eigs)

        idx = eigs.argsort()
        idx = idx[::-1]
        eigs = eigs[idx]

        eigv = eigv[:, idx]
        numpy.savetxt(eigenvalues_file, eigs)

        if numpy.where(numpy.imag(eigs) >= tol)[0].size != 0:  # We check if there are no negligible complex parts
            L = min(N, numpy.where(numpy.imag(eigs) >= tol)[0][0])
            Lmax = numpy.where(numpy.imag(eigs) >= tol)[0][0]
        else:
            L = min(N, Lmax)

        # Remove (negigible) complex parts
        eigs = numpy.real(eigs)

        if numpy.where(eigs <= 0)[0].size != 0:
            L = min(L, numpy.where(eigs <= 0)[0][0])  # we do not admit negative eigenvalues.
            Lmax = min(Lmax, numpy.where(eigs <= 0)[0][0])

        ratio = numpy.sum(eigs[0:L]) / numpy.sum(eigs[0:Lmax])

        for i in range(L):
            print "lambda_", i, " = ", eigs[i]
            if i == 0:
                p = self.snapshots_matrix.dot(D.dot(eigv[:, i]))
                p = numpy.squeeze(numpy.asarray(p))  # convert from an N_h x 1 matrix to an N_h vector
                p /= numpy.sqrt(eigs[i])
                Z = p.reshape(-1, 1)  # as column vector
            else:
                p = self.snapshots_matrix.dot(D.dot(eigv[:, i]))
                p = numpy.squeeze(numpy.asarray(p))  # convert from an N_h x 1 matrix to an N_h vector
                p /= numpy.sqrt(eigs[i])
                Z = numpy.hstack((Z, p.reshape(-1, 1)))  # add new basis functions as column vectors

        print "Choosen ", L, "basis of ", Lmax, " with ratio=", ratio

        return (Z, L)
