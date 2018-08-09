from itertools import chain
from six.moves import map, range
from firedrake import (Constant, Function, FunctionSpace, interpolate,
                       NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunctions,
                       VectorFunctionSpace, div, nabla_grad, dot, dx, grad, inner, VectorSpaceBasis,
                       CellVolume, split, MixedVectorSpaceBasis, as_vector, sqrt, MixedFunctionSpace, derivative)
from .util import *


class IncNavierStokesEnEq(object):
    def __init__(self, mesh, nu, rho, cp, k):
        self.verbose = True
        self.mesh = mesh
        self.dt = 0.001
        self.nu = nu
        self.rho = rho
        self.cp = cp
        self.k = k
        self.mu = self.nu * self.rho
        self.has_nullspace = False

        self.V = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Q = FunctionSpace(self.mesh, "CG", 1)
        self.S = FunctionSpace(self.mesh, "CG", 1)
        self.W = MixedFunctionSpace([self.V, self.Q, self.S])

        self.forcing = Function(self.V)

        self.solver_parameters = {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "fgmres",
            "pc_type": "asm",
            "pc_asm_type": "restrict",
            "pc_asm_overlap": 1,
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
        self.upT0 = Function(self.W)
        self.u0, self.p0, self.T0 = split(self.upT0)

        self.upT = Function(self.W)
        self.u1, self.p1, self.T1 = split(self.upT)

        self.upT.sub(0).rename("velocity")
        self.upT.sub(1).rename("pressure")
        self.upT.sub(2).rename("temperature")

        v, q, s = TestFunctions(self.W)

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
        self.F = (1.0 / self.dt) * inner(self.u1 - self.u0, v) * dx

        # weak form
        self.F += (
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
        self.F += tau * inner(
            + dot(self.u0, nabla_grad(v))
            - self.nu * div(grad(v))
            + (1.0 / self.rho) * grad(q), R) * dx

        self.F += (
                self.rho * self.cp * (1.0 / self.dt) * inner((self.T1 - self.T0), s) * dx
                + inner(dot(grad(self.T1), self.u1), s) * dx
                + self.k * inner(grad(self.T1), grad(s)) * dx
        )

        self.problem = NonlinearVariationalProblem(self.F, self.upT, self.bcs)
        self.solver = NonlinearVariationalSolver(
            self.problem,
            nullspace=nullspace,
            solver_parameters=self.solver_parameters)

    def get_mixed_fs(self):
        return self.W

    def set_forcing(self, f):
        self.forcing.project(f)

    def set_bcs(self, u_bcs, p_bcs, T_bcs):
        self.bcs = list(chain.from_iterable([u_bcs, p_bcs, T_bcs]))

    def step(self):
        if self.verbose:
            printp0("IncNavierStokesEnEq")
        self.solver.solve()
        self.upT0.assign(self.upT)
        return self.upT.split()
