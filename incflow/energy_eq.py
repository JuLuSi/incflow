from __future__ import absolute_import, division, print_function
from firedrake import (Constant, Function, FunctionSpace,
                       NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunction,
                       VectorFunctionSpace, dot, dx, grad, inner, nabla_grad, assemble, TrialFunction)
from .util import *


class EnergyEq(object):
    def __init__(self, mesh):
        self.verbose = True
        self.mesh = mesh
        self.dt = 0.001
        self.k = 0.0257
        self.rho = 1.0
        self.cp = 1.005
        self.S = FunctionSpace(self.mesh, "CG", 1)

        self.energy_eq_solver_parameters = {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "cg",
            "ksp_atol": 1e-10,
            "pc_type": "hypre",
        }

        if self.verbose:
            self.energy_eq_solver_parameters["snes_monitor"] = True
            self.energy_eq_solver_parameters["ksp_converged_reason"] = True

    def _weak_form(self, u, T, s, rho, cp, k):
        F = inner(dot(grad(T), u), s) * dx \
            + k * inner(grad(T), grad(s)) * dx
        return F

    def setup_solver(self, V):
        self.V = V
        self.u = Function(self.V)
        s = TestFunction(self.S)
        self.T_1 = Function(self.S)
        self.T0 = Function(self.S)
        self.T1 = Function(self.S)
        self.T1.rename("temperature")

        idt = Constant(self.dt)
        self.T_1.assign(self.T0)

        # ENERGY EQUATION

        # BACKWAD EULER

        F1 = inner((self.T1 - self.T0), s) * dx \
             + idt / (self.rho * self.cp) \
               * self._weak_form(self.u, self.T1, s, self.rho, self.cp, self.k)

        self.energy_eq_problem = NonlinearVariationalProblem(
            F1, self.T1, self.T_bcs)

        self.energy_eq_solver = NonlinearVariationalSolver(
            self.energy_eq_problem,
            solver_parameters=self.energy_eq_solver_parameters)

    def get_fs(self):
        return self.S

    def get_mass_matrix(self):
        s = TestFunction(self.S)
        t = TrialFunction(self.S)
        M = inner(t, s) * dx

        return assemble(M)

    def set_bcs(self, T_bcs):
        self.T_bcs = T_bcs

    def set_u(self, u):
        self.u.assign(u)

    def step(self, u):
        self.set_u(u)
        if self.verbose:
            printp0("EnergyEquation")

        self.energy_eq_solver.solve()
        self.T_1.assign(self.T0)
        self.T0.assign(self.T1)

        return self.T1
