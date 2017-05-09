from __future__ import absolute_import, division, print_function
from firedrake import (Constant, Function, FunctionSpace,
                       NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunction,
                       VectorFunctionSpace, dot, dx, grad, inner, nabla_grad)
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

        # Temporal discretization: F-EULER, B-EULER, CRANIC, BDF2
        self.temp_disc = "BDF2"

        self.energy_eq_solver_parameters = {
            "mat_type": "aij",
            "ksp_type": "preonly",
            "ksp_atol": 1e-8,
            "pc_type": "lu",
            "pc_factor_mat_solver_package": "mumps"
        }

        if self.verbose:
            self.energy_eq_solver_parameters["snes_monitor"] = True

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

        # FORWARD EULER
        if self.temp_disc == "F-EULER":
            #printp0("Temporal Difference: Forward-Euler")
            F1 = inner((self.T1 - self.T0), s) * dx \
                + idt / (self.rho * self.cp) \
                * self._weak_form(self.u, self.T0, s, self.rho, self.cp, self.k)

        # BACKWAD EULER
        elif self.temp_disc == "B-EULER":
            #printp0("Temporal Difference: Backward-Euler")
            F1 = inner((self.T1 - self.T0), s) * dx \
                + idt / (self.rho * self.cp) \
                * self._weak_form(self.u, self.T1, s, self.rho, self.cp, self.k)

        # CRANK NICOLSON
        elif self.temp_disc == "CRANIC":
            #printp0("Temporal Difference: Crank-Nicolson")
            F1 = inner((self.T1 - self.T0), s) * dx \
                + idt / (self.rho * self.cp) * 0.5 * (
                self._weak_form(self.u, self.T0, s, self.rho, self.cp, self.k) +
                self._weak_form(self.u, self.T1, s, self.rho, self.cp, self.k))

        # BDF2
        elif self.temp_disc == "BDF2":
            #printp0("Temporal Difference: BDF2")
            F1 = inner(1.5 * self.T1 - 2.0 * self.T0 + 0.5 * self.T_1, s) * dx \
                + idt / (self.rho * self.cp) \
                * self._weak_form(self.u, self.T1, s, self.rho, self.cp, self.k)

        self.energy_eq_problem = NonlinearVariationalProblem(
            F1, self.T1, self.T_bcs)

        self.energy_eq_solver = NonlinearVariationalSolver(
            self.energy_eq_problem,
            solver_parameters=self.energy_eq_solver_parameters)

    def get_T_fs(self):
        return self.S

    def set_bcs(self, T_bcs):
        self.T_bcs = T_bcs

    def set_u(self, u):
        self.u.assign(u)

    def step(self, u):
        self.set_u(u)
        if self.verbose:
            printp0("EnergyEquations")

        self.energy_eq_solver.solve()
        self.T_1.assign(self.T0)
        self.T0.assign(self.T1)

        return self.T1
