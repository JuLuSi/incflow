from __future__ import absolute_import, division, print_function
from itertools import chain
from six.moves import map, range
from firedrake import (Constant, Function, FunctionSpace, interpolate,
                       NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunctions,
                       VectorFunctionSpace, div, nabla_grad, dot, dx, grad, inner, VectorSpaceBasis,
                       CellVolume, split, MixedVectorSpaceBasis, as_vector, sqrt)
from .util import *


class IncNavierStokes(object):
    def __init__(self, mesh, nu, rho):
        self.verbose = True
        self.mesh = mesh
        self.dt = 0.001
        self.nu = nu
        self.rho = rho
        self.mu = self.nu * self.rho
        self.has_nullspace = False

        self.forcing = Constant((0.0, 0.0))

        self.V = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Q = FunctionSpace(self.mesh, "CG", 1)
        self.W = self.V * self.Q

        self.solver_parameters = {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "fgmres",
            "pc_type": "asm",
            "pc_asm_type": "restrict",
            "pc_asm_overlap": 2,
            "sub_ksp_type": "preonly",
            "sub_pc_type": "ilu",
            "sub_pc_factor_levels": 1,
        }

        if self.verbose:
            self.solver_parameters["snes_monitor"] = True
            self.solver_parameters["ksp_converged_reason"] = True

    def setup_solver(self):
        """ Setup the solvers
        """
        self.up0 = Function(self.W)
        self.u0, self.p0 = split(self.up0)

        self.up = Function(self.W)
        self.u1, self.p1 = split(self.up)

        self.up.sub(0).rename("velocity")
        self.up.sub(1).rename("pressure")

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
            + inner(dot(self.u0, nabla_grad(self.u1)), v) * dx
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

        # GLS
        F += tau * inner(
            + dot(self.u0, nabla_grad(v))
            - self.nu * div(grad(v))
            + (1.0 / self.rho) * grad(q), R) * dx

        self.problem = NonlinearVariationalProblem(F, self.up, self.bcs)
        self.solver = NonlinearVariationalSolver(
            self.problem,
            nullspace=nullspace,
            solver_parameters=self.solver_parameters)

    def get_mixed_fs(self):
        return self.W

    def set_forcing(self, forcing):
        self.forcing = forcing

    def set_bcs(self, u_bcs, p_bcs):
        self.bcs = list(chain.from_iterable([u_bcs, p_bcs]))

    def step(self):
        if self.verbose:
            printp0("IncNavierStokes")
        self.solver.solve()
        self.up0.assign(self.up)
        return self.up.split()
