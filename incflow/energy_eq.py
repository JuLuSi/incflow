from firedrake import (Constant, Function, FunctionSpace,
                       NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunction,
                       VectorFunctionSpace, dot, dx, grad, inner, nabla_grad, assemble, TrialFunction)
from .util import *


class EnergyEq(object):
    def __init__(self, mesh, rho, k, cp):
        self.verbose = True
        self.mesh = mesh
        self.dt = 0.001
        self.k = k  # 0.0257
        self.rho = rho  # 1.1644
        self.cp = cp  # 1005
        self.S = FunctionSpace(self.mesh, "CG", 1)

        self.energy_eq_solver_parameters = {
            "mat_type": "aij",
            "ksp_type": "fgmres",
            "pc_type": "asm",
            "pc_asm_type": "restrict",
            "pc_asm_overlap": 1,
            "sub_ksp_type": "preonly",
            "sub_pc_type": "ilu",
            "sub_pc_factor_levels": 1,
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

        self.idt = Constant(self.dt)
        self.T_1.assign(self.T0)

        # ENERGY EQUATION

        # BACKWAD EULER

        F = (
                (1.0 / self.dt) * self.rho * self.cp * inner((self.T1 - self.T0), s) * dx
                + self.rho * self.cp * inner(dot(self.u, grad(self.T1)), s) * dx
                + self.k * inner(grad(self.T1), grad(s)) * dx
        )

        self.energy_eq_problem = NonlinearVariationalProblem(
            F, self.T1, self.T_bcs)

        self.energy_eq_solver = NonlinearVariationalSolver(
            self.energy_eq_problem,
            solver_parameters=self.energy_eq_solver_parameters)

    def get_fs(self):
        return self.S

    def get_mass_matrix(self):
        s = TestFunction(self.S)
        t = TrialFunction(self.S)
        M = inner(t, s) * dx

        return M

    def get_H1_matrix(self):
        s = TestFunction(self.S)
        t = TrialFunction(self.S)
        H = inner(t, s) * dx + inner(grad(t), grad(s)) * dx

        return H

    def get_jacobian_matrix(self):
        return self.energy_eq_problem.J

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
