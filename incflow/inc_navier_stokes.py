from __future__ import absolute_import, division, print_function
from six.moves import map, range
from firedrake import (Constant, Function, FunctionSpace, CellSize, sqrt, VectorSpaceBasis,
                       NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunction,
                       VectorFunctionSpace, div, dot, dx, grad, inner)
from .util import *


class IncNavierStokes(object):
    def __init__(self, mesh):
        self.verbose = True
        self.mesh = mesh
        self.dt = 0.001
        self.nu = 0.001
        self.rho = 1.0
        self.mu = self.nu * self.rho

        self.pressure_nullspace = False
        self.supg = False

        self.time_integration_method = "backward_euler"

        self.forcing = Constant((0.0, 0.0))

        self.V = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Q = FunctionSpace(self.mesh, "CG", 1)

        self.tent_vel_solver_parameters = {
            "mat_type": "aij",
            "snes_atol": 1.0e-10,
        }
        self.pressure_poisson_solver_parameters = {
            "mat_type": "aij",
            "snes_atol": 1.0e-10,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_boomeramg_relax_type_coarse": "jacobi",
            "pc_hypre_boomeramg_strong_threshold": 0.25,
            "pc_hypre_boomeramg_agg_nl": 2,
            "pc_hypre_boomeramg_max_levels": 25
        }
        self.velocity_corr_solver_parameters = {
            "mat_type": "aij",
            "snes_atol": 1.0e-10,
            "ksp_type": "cg",
            "pc_type": "hypre"
        }
        if self.verbose:
            self.tent_vel_solver_parameters["snes_monitor"] = True
            # self.tent_vel_solver_parameters["ksp_monitor"] = True
            self.pressure_poisson_solver_parameters["snes_monitor"] = True
            # self.pressure_poisson_solver_parameters["ksp_monitor"] = True
            self.velocity_corr_solver_parameters["snes_monitor"] = True

    def _weak_form(self, u, v, p, f, rho, mu):
        F = (
            mu * inner(grad(u), grad(v)) * dx
            + rho * inner(grad(u) * u, v) * dx
            # + rho * 0.5 * (inner(grad(u)*u, v) - inner(grad(v)*u, u)) * dx
            + inner(grad(p), v) * dx
            - inner(f, v) * dx
        )
        return F

    def _residual_form(self, u, p, f, rho, mu):
        R = (
            - mu * div(grad(u))
            + rho * grad(u) * u
            + grad(p)
            - f
        )
        return R

    def setup_solver(self):
        """ Setup the solvers

        For details look at
        J.L. Guermond, P. Minev, Jie Chen "An overview of projection methods for
        incompressible flows"
        """
        v = TestFunction(self.V)
        ui = Function(self.V)
        self.u_1 = Function(self.V)
        self.u0 = Function(self.V)
        self.u1 = Function(self.V)
        self.u1.rename("velocity")
        q = TestFunction(self.Q)
        self.p0 = Function(self.Q)
        self.p1 = Function(self.Q)
        self.p1.rename("pressure")

        printp0('******')
        printp0('Incompressible Navier Stokes')
        printp0('Simulation info')
        printp0('DOFs: {}'.format(self.u0.vector().size()))
        printp0('******')

        k = Constant(self.dt)

        if self.supg:
            h = CellSize(self.mesh)
            u_norm = sqrt(dot(self.u0, self.u0))
            tau = ((2.0 / self.dt)**2 + (2.0 * u_norm / h)
                   ** 2 + (4.0 * self.nu / h**2)**2)**(-0.5)

        # ui.assign(self.u0)
        self.u_1.assign(self.u0)

        if self.verbose:
            printp0("Time integration: {}".format(
                self.time_integration_method))

        # Tentative velocity step
        if self.time_integration_method == "forward_euler":
            F1 = inner(ui - self.u0, v) * dx \
                + (k / self.rho) * self._weak_form(self.u0, v,
                                                   self.p0, self.forcing, self.rho, self.mu)

        if self.time_integration_method == "backward_euler":
            F1 = inner(ui - self.u0, v) * dx \
                + (k / self.rho) * self._weak_form(ui, v,
                                                   self.p0, self.forcing, self.rho, self.mu)
            if self.supg:
                R1 = (ui - self.u0) + (k / self.rho) * self._residual_form(ui,
                                                                           self.p0, self.forcing, self.rho, self.mu)
                F1 += tau * inner(grad(v) * ui, R1) * dx
                # F1 += tau * 1.0 / self.rho * inner(grad(q), R1) * dx

        if self.time_integration_method == "crank_nicolson":
            F1 = inner(ui - self.u0, v) * dx \
                + (k / self.rho) * 0.5 * (
                    self._weak_form(self.u0, v, self.p0, self.forcing, self.rho, self.mu) +
                    self._weak_form(ui, v, self.p0, self.forcing, self.rho, self.mu))

        if self.time_integration_method == "bdf2":
            F1 = inner(1.5 * ui - 2.0 * self.u0 + 0.5 * self.u_1, v) * dx \
                + (k / self.rho) * self._weak_form(ui, v,
                                                   self.p0, self.forcing, self.rho, self.mu)

        self.tent_vel_problem = NonlinearVariationalProblem(F1, ui, self.u_bcs)
        self.tent_vel_solver = NonlinearVariationalSolver(
            self.tent_vel_problem,
            options_prefix="tvel_",
            solver_parameters=self.tent_vel_solver_parameters)

        # Pressure correction
        if self.time_integration_method == "bdf2":
            F2 = dot(grad(self.p1), grad(q)) * dx \
                - dot(grad(self.p0), grad(q)) * dx \
                + ((3.0 * self.rho) / (2.0 * k)) * div(ui) * q * dx
        else:
            F2 = dot(grad(self.p1), grad(q)) * dx \
                - dot(grad(self.p0), grad(q)) * dx \
                + (self.rho / k) * div(ui) * q * dx

        nullspace = None
        if self.pressure_nullspace:
            null_vec = Function(self.Q)
            null_vec.assign(0.0)
            nullspace = VectorSpaceBasis(constant=True)

        self.pressure_poisson_problem = NonlinearVariationalProblem(
            F2, self.p1, self.p_bcs)
        self.pressure_poisson_solver = NonlinearVariationalSolver(
            self.pressure_poisson_problem,
            options_prefix="prep_",
            nullspace=nullspace,
            solver_parameters=self.pressure_poisson_solver_parameters)

        # Velocity correction
        phi = self.p1 - self.p0
        if self.time_integration_method == "bdf2":
            F3 = inner(3.0 * self.u1 - 3.0 * ui, v) * dx \
                + (2.0 * k / self.rho) * inner(grad(phi), v) * dx
        else:
            F3 = inner(self.u1 - ui, v) * dx \
                + (k / self.rho) * inner(grad(phi), v) * dx

        self.velocity_corr_problem = NonlinearVariationalProblem(
            F3, self.u1, self.u_bcs)
        self.velocity_corr_solver = NonlinearVariationalSolver(
            self.velocity_corr_problem,
            options_prefix="velc_",
            solver_parameters=self.velocity_corr_solver_parameters)

    def get_u_fs(self):
        return self.V

    def get_p_fs(self):
        return self.Q

    def set_forcing(self, forcing):
        self.forcing = forcing

    def set_bcs(self, u_bcs, p_bcs):
        self.u_bcs = u_bcs
        self.p_bcs = p_bcs

    def step(self):
        if self.verbose:
            printp0("IncNavierStokes")

        if self.verbose:
            printp0("Tentative velocity step")
        self.tent_vel_solver.solve()

        if self.verbose:
            printp0("Pressure correction")
        self.pressure_poisson_solver.solve()

        if self.verbose:
            printp0("Velocity correction")
        self.velocity_corr_solver.solve()

        self.u_1.assign(self.u0)
        self.u0.assign(self.u1)
        self.p0.assign(self.p1)

        return self.u1, self.p1
