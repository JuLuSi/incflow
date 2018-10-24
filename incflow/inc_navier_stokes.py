from itertools import chain
from firedrake import (Constant, Function, FunctionSpace, interpolate,
                       NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunctions,
                       VectorFunctionSpace, div, nabla_grad, dot, dx, grad, inner, VectorSpaceBasis,
                       CellVolume, split, MixedVectorSpaceBasis, as_vector, sqrt)
from .util import *


class IncNavierStokes(object):
    def __init__(self, mesh, nu, rho, solver_preset='lu'):
        self.verbose = True
        self.mesh = mesh
        self.dt = 0.001
        self.nu = nu
        self.rho = rho
        self.mu = self.nu * self.rho
        self.has_nullspace = False

        self.V = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Q = FunctionSpace(self.mesh, "CG", 1)
        self.W = self.V * self.Q

        self.forcing = Function(self.V)

        if solver_preset == 'lu':
            self.solver_parameters = {
                "mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "gmres",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }

        elif solver_preset == 'asm':
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

        elif solver_preset == 'lsc_amg':
            self.solver_parameters = {
                "mat_type": "aij",
                "snes_type": "ksponly",
                "ksp_type": "fgmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "diag",
                "pc_fieldsplit_schur_precondition": "self",
                "fieldsplit_0": {
                    "ksp_type": "preonly",
                    "pc_type": "hypre",
                    "pc_hypre_boomeramg_print_statistics": False,
                    "pc_hypre_boomeramg_smooth_type": "Euclid",
                    "pc_hypre_boomeramg_P_max": 4,
                    "pc_hypre_boomeramg_agg_nl": 2,
                    "pc_hypre_boomeramg_no_CF": True,
                    "pc_hypre_boomeramg_agg_num_paths": 2,
                    "pc_hypre_boomeramg_coarsen_type": "HMIS",
                    "pc_hypre_boomeramg_interp_type": "ext+i",
                },
                "fieldsplit_1": {
                    "ksp_type": "preonly",
                    "pc_type": "lsc",
                    "lsc": {
                        "pc_type": "hypre",
                        # "pc_hypre_boomeramg_print_statistics": False,
                        # "pc_hypre_boomeramg_P_max": 4,
                        # "pc_hypre_boomeramg_agg_nl": 1,
                        # "pc_hypre_boomeramg_no_CF": True,
                        # "pc_hypre_boomeramg_agg_num_paths": 2,
                        # "pc_hypre_boomeramg_coarsen_type": "HMIS",
                        # "pc_hypre_boomeramg_interp_type": "ext+i",
                },
                },
            }

        if self.verbose:
            self.solver_parameters["snes_monitor"] = True
            self.solver_parameters["ksp_converged_reason"] = True
            self.solver_parameters["ksp_monitor"] = True

    def setup_solver(self):
        """ Setup the solvers
        """
        self.up0 = Function(self.W)
        self.u0, self.p0 = split(self.up0)

        self.up1 = Function(self.W)
        self.u1, self.p1 = split(self.up1)

        self.up1.sub(0).rename("velocity")
        self.up1.sub(1).rename("pressure")

        v, q = TestFunctions(self.W)

        h = CellVolume(self.mesh)
        u_norm = sqrt(dot(self.u0, self.u0))

        if self.has_nullspace:
            nullspace = MixedVectorSpaceBasis(
                self.W, [self.W.sub(0), VectorSpaceBasis(constant=True)])
        else:
            nullspace = None

        tau = ((2.0 / self.dt) ** 2 + (2.0 * u_norm / h)
               ** 2 + (4.0 * self.nu / h ** 2) ** 2) ** (-0.5)

        # temporal discretization
        F = (1.0 / self.dt) * inner(self.u1 - self.u0, v) * dx

        # weak form
        F += (
            + inner(dot(self.u1, nabla_grad(self.u1)), v) * dx
            + self.nu * inner(grad(self.u1), grad(v)) * dx
            - (1.0 / self.rho) * self.p1 * div(v) * dx
            + div(self.u1) * q * dx
            - inner(self.forcing, v) * dx
        )

        # residual form
        R = (
            + (1.0 / self.dt) * (self.u1 - self.u0)
            + dot(self.u0, nabla_grad(self.u1))
            - self.nu * div(grad(self.u1))
            + (1.0 / self.rho) * grad(self.p1)
            - self.forcing
        )

        # SUPG
        # F += inner(tau * dot(self.u0, nabla_grad(v)), R) * self.dx

        # PSPG
        # F += inner(1.0 / self.rho * tau * nabla_grad(q), R) * dx

        self.problem = NonlinearVariationalProblem(F, self.up1, self.bcs)
        self.solver = NonlinearVariationalSolver(
            self.problem,
            options_prefix='incns',
            nullspace=nullspace,
            solver_parameters=self.solver_parameters)

    def get_mixed_fs(self):
        return self.W

    def set_forcing(self, f):
        self.forcing.project(f)

    def set_bcs(self, u_bcs, p_bcs):
        self.bcs = list(chain.from_iterable([u_bcs, p_bcs]))

    def step(self):
        if self.verbose:
            printp0("IncNavierStokes")
        self.solver.solve()
        self.up0.assign(self.up1)
        return self.up1.split()
