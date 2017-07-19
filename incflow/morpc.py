from __future__ import absolute_import, print_function
from firedrake import PCBase
from firedrake.assemble import allocate_matrix, create_assembly_callable
from firedrake.petsc import PETSc


class MORPC(PCBase):
    def initialize(self, pc):
        _, K = pc.getOperators()
        self.ctx = K.getPythonContext()
        self.prefix = pc.getOptionsPrefix()

        self.assembleK()

        # Reassign K
        K = self.K.M.handle

        self.Z = self.ctx.appctx.get("projection_mat", None)
        assert self.Z != None

        # Project the jacobian
        Kp = K.PtAP(self.Z)

        KpInv = PETSc.KSP().create()
        KpInv.setOperators(Kp)
        #KpInv.setType("preonly")
        KpInvPC = KpInv.getPC()
        KpInvPC.setFactorSolverPackage("mumps")
        KpInv.setUp()
        self.KpInv = KpInv

        # Create projected space vectors
        self.Xp, _ = self.Z.createVecs()
        self.Yp, _ = self.Z.createVecs()

    def update(self, pc):
        pass

    def apply(self, pc, X, Y):
        # Project Y and Y into the projected space
        self.Z.multTranspose(Y, self.Yp)
        self.Z.multTranspose(X, self.Xp)

        # Solve
        self.KpInv.solve(self.Xp, self.Yp)

        # Project x back to original space
        self.Z.mult(self.Xp, X)
        self.Z.mult(self.Yp, Y)

    def applyTranspose(self, pc, X, Y):
        pass

    def assembleK(self):
        ctx = self.ctx
        mat_type = PETSc.Options().getString(self.prefix + "assembled_mat_type", "aij")

        self.K = allocate_matrix(ctx.a, bcs=ctx.row_bcs,
                                 form_compiler_parameters=ctx.fc_params,
                                 mat_type=mat_type)

        self._assemble_K = create_assembly_callable(ctx.a, tensor=self.K,
                                                    bcs=ctx.row_bcs,
                                                    form_compiler_parameters=ctx.fc_params,
                                                    mat_type=mat_type)
        self._assemble_K()

        self.mat_type = mat_type

        self.K.force_evaluation()
