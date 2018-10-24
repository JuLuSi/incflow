from itertools import chain
from six.moves import map, range
from firedrake import (Constant, Function, FunctionSpace, interpolate,
                       NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunctions,
                       VectorFunctionSpace, div, nabla_grad, dot, dx, grad, inner, VectorSpaceBasis,
                       CellVolume, split, MixedVectorSpaceBasis, as_vector, sqrt, MixedFunctionSpace, derivative, projection)
from .util import *


class IncNavierStokesEnEq(object):
    def __init__(self, mesh, nu, rho, cp, k):
        self.verbose = True
        self.mesh = mesh
        self.dt = Constant(0.001)
        self.nu = nu
        self.rho = rho
        self.cp = cp
        self.k = k
        self.mu = self.nu * self.rho
        self.has_nullspace = False
        self.time_discretization = 'BDF1'
        self.td_theta = 0.5

        P = 2
        self.V = VectorFunctionSpace(self.mesh, "CG", P)
        self.Q = FunctionSpace(self.mesh, "CG", P-1)
        self.S = FunctionSpace(self.mesh, "CG", 1)
        self.W = MixedFunctionSpace([self.V, self.Q, self.S])

        self.forcing = Function(self.W.sub(0))

        self.solver_parameters = {
            "mat_type": "aij",
            "ksp_type": "fgmres",
            "pc_type": "asm",
            "pc_asm_type": "restrict",
            "pc_asm_overlap": 1,
            "sub_ksp_type": "preonly",
            "sub_pc_type": "ilu",
            "sub_pc_factor_levels": 1,
        }

        # self.solver_parameters = {
        #     "mat_type": "aij",
        #     "ksp_type": "fgmres",
        #     "ksp_gmres_restart": 300,
        #     "pc_type": "fieldsplit",
        #     "pc_fieldsplit_0_fields": "0,1",
        #     "pc_fieldsplit_1_fields": "2",
        #     "fieldsplit_0": {
        #         "ksp_type": "gmres",
        #         "ksp_rtol": 1e-4,
        #         "pc_type": "fieldsplit",
        #         "pc_fieldsplit_type": "schur",
        #         "pc_fieldsplit_schur_fact_type": "diag",
        #         "pc_fieldsplit_schur_precondition": "self",
        #         "fieldsplit_0": {
        #             "ksp_type": "preonly",
        #             "pc_type": "hypre",
        #             "pc_hypre_boomeramg_print_statistics": False,
        #             "pc_hypre_boomeramg_smooth_type": "Euclid",
        #             "pc_hypre_boomeramg_P_max": 4,
        #             "pc_hypre_boomeramg_agg_nl": 1,
        #             "pc_hypre_boomeramg_no_CF": True,
        #             "pc_hypre_boomeramg_agg_num_paths": 2,
        #             "pc_hypre_boomeramg_coarsen_type": "HMIS",
        #             "pc_hypre_boomeramg_interp_type": "ext+i",
        #         },
        #         "fieldsplit_1": {
        #             "ksp_type": "preonly",
        #             "pc_type": "lsc",
        #             "lsc": {
        #                 "pc_type": "hypre",
        #                 # "pc_hypre_boomeramg_print_statistics": False,
        #                 # "pc_hypre_boomeramg_P_max": 4,
        #                 # "pc_hypre_boomeramg_agg_nl": 1,
        #                 # "pc_hypre_boomeramg_no_CF": True,
        #                 # "pc_hypre_boomeramg_agg_num_paths": 2,
        #                 # "pc_hypre_boomeramg_coarsen_type": "HMIS",
        #                 # "pc_hypre_boomeramg_interp_type": "ext+i",
        #             },
        #         },
        #     },
        #     "fieldsplit_1": {
        #         "ksp_type": "gmres",
        #         "ksp_rtol": 1e-4,
        #         "pc_type": "hypre",
        #         "pc_hypre_boomeramg_print_statistics": False,
        #         "pc_hypre_boomeramg_smooth_type": "Euclid",
        #         "pc_hypre_boomeramg_P_max": 4,
        #         "pc_hypre_boomeramg_agg_nl": 1,
        #         "pc_hypre_boomeramg_agg_num_path": 2,
        #         "pc_hypre_boomeramg_coarsen_type": "HMIS",
        #         "pc_hypre_boomeramg_interp_type": "ext+i",
        #         "pc_hypre_boomeramg_no_CF": True,
        #     },
        # }

        if self.verbose:
            self.solver_parameters["snes_monitor"] = True
            self.solver_parameters["snes_converged_reason"] = True
            self.solver_parameters["ksp_converged_reason"] = True

    def setup_solver(self, stabilization=False):
        """ Setup the solvers
        """
        self.upT0 = Function(self.W)
        self.u0, self.p0, self.T0 = split(self.upT0)

        self.upT1 = Function(self.W)
        self.u1, self.p1, self.T1 = split(self.upT1)

        self.upT2 = Function(self.W)
        self.u2, self.p2, self.T2 = split(self.upT2)

        self.upTmr = None  # Most recent timestep

        for block in [self.upT0, self.upT1, self.upT2]:
            block.sub(0).rename("velocity")
            block.sub(1).rename("pressure")
            block.sub(2).rename("temperature")

        v, q, s = TestFunctions(self.W)

        h = CellVolume(self.mesh)
        u_norm = sqrt(dot(self.u0, self.u0))

        if self.has_nullspace:
            nullspace = MixedVectorSpaceBasis(
                self.W, [self.W.sub(0), VectorSpaceBasis(constant=True), self.W.sub(2)])
        else:
            nullspace = None

        tau = ((2.0 / self.dt) ** 2 + (2.0 * u_norm / h)
               ** 2 + (4.0 * self.nu / h ** 2) ** 2) ** (-0.5)

        def ins_form(u, p, T):
            F = (
                    + inner(dot(u, nabla_grad(u)), v) * dx
                    + self.nu * inner(grad(u), grad(v)) * dx
                    - (1.0 / self.rho) * p * div(v) * dx
                    + div(u) * q * dx
                    - inner(self.forcing, v) * dx
            )
            return F

        def eneq_form(u, T):
            F = (
                    + self.rho * self.cp * inner(dot(u, grad(T)), s) * dx
                    + self.k * inner(grad(T), grad(s)) * dx
            )
            return F

        if self.time_discretization == 'BDF1':
            self.upTmr = self.upT1

            self.F = 1.0 / self.dt * inner(self.u1 - self.u0, v) * dx
            self.F += ins_form(self.u1, self.p1, self.T1)

            self.F += 1.0 / self.dt * self.rho * self.cp * inner((self.T1 - self.T0), s) * dx
            self.F += eneq_form(self.u1, self.T1)
        elif self.time_discretization == 'BDF2':
            self.upTmr = self.upT2

            self.F = Constant(1.0 / (2.0 * self.dt)) * inner(3.0 * self.u2 - 4.0 * self.u1 + self.u0, v) * dx
            self.F += ins_form(self.u2, self.p2, self.T2)

            self.F += Constant(1.0 / (2.0 * self.dt)) * inner(3.0 * self.T2 - 4.0 * self.T1 + self.T0, s) * dx
            self.F += eneq_form(self.u2, self.T2)
        elif self.time_discretization == 'Theta':
            self.upTmr = self.upT1

            self.F = Constant(1.0 / self.dt) * inner(self.u1 - self.u0, v) * dx
            self.F += Constant(self.td_theta) * ins_form(self.u1, self.p1, self.T1)
            self.F += Constant(1.0 - self.td_theta) * ins_form(self.u0, self.p0, self.T0)

            self.F += Constant(1.0 / self.dt) * self.rho * self.cp * inner((self.T1 - self.T0), s) * dx
            self.F += Constant(self.td_theta) * eneq_form(self.u1, self.T1)
            self.F += Constant(1.0 - self.td_theta) * eneq_form(self.u0, self.T0)

            # if stabilization:
        #     # GLS
        #
        #     # residual form
        #     R = (
        #             + (1.0 / self.dt) * (self.u1 - self.u0)
        #             + dot(self.u0, nabla_grad(self.u1))
        #             - self.nu * div(grad(self.u1))
        #             + (1.0 / self.rho) * grad(self.p1)
        #             - self.forcing
        #     )
        #
        #     self.F += tau * inner(
        #         + dot(self.u0, nabla_grad(v))
        #         - self.nu * div(grad(v))
        #         + (1.0 / self.rho) * grad(q), R) * dx

        self.problem = NonlinearVariationalProblem(self.F, self.upTmr, self.bcs)
        self.solver = NonlinearVariationalSolver(
            self.problem,
            options_prefix='coupled',
            nullspace=nullspace,
            solver_parameters=self.solver_parameters)

    def get_mixed_fs(self):
        return self.W

    def build_forcing_projector(self, f):
       	self.forcing_projector = projection.Projector(f, self.forcing)

    def set_forcing(self):
        self.forcing_projector.project()

    def set_bcs(self, u_bcs, p_bcs, T_bcs):
        self.bcs = list(chain.from_iterable([u_bcs, p_bcs, T_bcs]))

    def solve(self):
        if self.verbose:
            printp0("IncNavierStokesEnEq")
        self.solver.solve()

    def step(self):
        if self.time_discretization == 'BDF1':
            self.upT0.assign(self.upT1)
        elif self.time_discretization == 'BDF2':
            self.upT0.assign(self.upT1)
            self.upT1.assign(self.upT2)
        elif self.time_discretization == 'Theta':
            self.upT0.assign(self.upT1)

        return self.upTmr.split()
