import numpy as np
from abc import ABC, abstractmethod


class LineSearch(ABC):
    def __init__(self, f, x0, obj_tol, param_tol, max_iter):
        self.f = f
        self.x0 = x0
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.current_iteration = 0
        self.current_x = x0.copy()
        self.current_fx = self.f.objective(x0)
        self.history = [(x0.copy(), self.current_fx)]
        self.success = False
        self.output_message = ""

    def backtracking(self, x, d, fx, gx, c1=0.01, rho=0.5, max_backtrack=50):
        alpha = 1.0
        for _ in range(max_backtrack):
            new_x = x + alpha * d
            new_fx = self.f.objective(new_x)
            if new_fx <= fx + c1 * alpha * np.dot(gx, d):
                return alpha
            alpha *= rho
        return alpha

    @abstractmethod
    def step(self):
        pass

    def minimize(self):
        while self.step():
            pass
        return self.current_x, self.current_fx, self.success, self.history


class GradientDescent(LineSearch):
    def step(self):
        if self.current_iteration >= self.max_iter:
            self.success = False
            self.output_message = "Maximum iterations reached."
            return False
        x = self.current_x
        fx = self.current_fx
        g = self.f.gradient(x)
        d = -g  # Descent direction
        alpha = self.backtracking(x, d, fx, g)
        new_x = x + alpha * d
        new_fx = self.f.objective(new_x)
        # Check convergence
        if abs(new_fx - fx) < self.obj_tol:
            self.success = True
            self.output_message = "Objective tolerance met."
            return False
        if np.linalg.norm(new_x - x) < self.param_tol:
            self.success = True
            self.output_message = "Parameter tolerance met."
            return False
        # Update state
        self.current_x = new_x
        self.current_fx = new_fx
        self.history.append((new_x.copy(), new_fx))
        self.current_iteration += 1
        return True


class NewtonMethod(LineSearch):
    def step(self):
        if self.current_iteration >= self.max_iter:
            self.success = False
            self.output_message = "Maximum iterations reached."
            return False
        x = self.current_x
        fx = self.current_fx
        g = self.f.gradient(x)
        H = self.f.hessian(x)
        try:
            d = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            d = -g
        if np.dot(g, d) >= 0:
            d = -g
        alpha = self.backtracking(x, d, fx, g)
        new_x = x + alpha * d
        new_fx = self.f.objective(new_x)
        if abs(new_fx - fx) < self.obj_tol:
            self.success = True
            self.output_message = "Objective tolerance met."
            return False
        if np.linalg.norm(new_x - x) < self.param_tol:
            self.success = True
            self.output_message = "Parameter tolerance met."
            return False
        self.current_x = new_x
        self.current_fx = new_fx
        self.history.append((new_x.copy(), new_fx))
        self.current_iteration += 1
        return True
