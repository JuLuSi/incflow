from __future__ import absolute_import, division, print_function
from firedrake import (Constant, Function, FunctionSpace,
                       NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunction,
                       VectorFunctionSpace, dot, dx, grad, inner, nabla_grad, assemble, TrialFunction)
from incflow.util import *


class EnergyEqAdj(object):
    def __init__(self, mesh):
        self.verbose = True
        self.mesh = mesh
        self.dt = 0.001
        self.k = 0.0257
        self.rho = 1.0
        self.cp = 1.005
        self.alpha = 1.0
        self.S = FunctionSpace(self.mesh, "CG", 1)
        self.T = Function(self.S)
        self.Td = Function(self.S)

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

    def _weak_form(self, u, q, s, rho, cp, k):
        F = inner(dot(grad(s), u), q) * dx \
            + k * inner(grad(q), grad(s)) * dx
        return F

    def setup_solver(self, V):
        self.V = V
        self.u = Function(self.V)
        s = TestFunction(self.S)
        self.q_0 = Function(self.S)
        self.q0 = Function(self.S)
        self.q1 = Function(self.S)
        self.q0.rename("adjoint_variable")
        self.q1.rename("adjoint_variable")

        self.idt = Constant(self.dt)
        self.q1.assign(self.alpha * (self.Td - self.T))
        self.q_0.assign(self.q1)

        # ENERGY EQUATION ADJOINT

        # BACKWAD EULER

        F1 = -inner((self.q1 - self.q0), s) * dx \
             + self.idt / (self.rho * self.cp) * self._weak_form(self.u, self.q0, s, self.rho, self.cp, self.k) \
             - self.alpha * inner((self.Td - self.T), s) * dx

        self.energy_eq_problem = NonlinearVariationalProblem(
            F1, self.q0, self.q_bcs)

        self.energy_eq_solver = NonlinearVariationalSolver(
            self.energy_eq_problem,
            solver_parameters=self.energy_eq_solver_parameters)

    def get_fs(self):
        return self.S

    def get_jacobian_matrix(self):
        return self.energy_eq_problem.J

    def set_bcs(self, q_bcs):
        self.q_bcs = q_bcs

    def set_u(self, u):
        self.u.assign(u)

    def set_T(self, T):
        self.T.assign(T)

    def set_Td(self, Td):
        self.Td.assign(Td)

    def step(self, u, T):
        self.set_u(u)
        self.set_T(T)
        if self.verbose:
            printp0("EnergyEquationAdjoint")

        self.energy_eq_solver.solve()
        self.q_0.assign(self.q1)
        self.q1.assign(self.q0)

        return self.q0
