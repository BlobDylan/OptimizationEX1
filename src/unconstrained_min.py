import numpy as np
from tests.examples import Function
from abc import ABC, abstractmethod


class LineSearch:
    def __init__(self, f: Function, x0, obj_tol, param_tol, max_iter):
        self.f = f
        self.x0 = x0
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter

        self.current_iteration = 0
        self.current_x = x0
        self.current_fx = f.objective(x0)
        self.history = [(x0, self.current_fx)]

        self.success = False
        self.output_message = None

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def minimize(self):
        pass


class GradientDescent(LineSearch):
    def __init__(self, f: Function, x0, obj_tol, param_tol, max_iter, step_size):
        super().__init__(f, x0, obj_tol, param_tol, max_iter)
        self.step_size = step_size

    def step(self):
        if self.current_iteration >= self.max_iter:
            self.success = False
            self.output_message = "Maximum iterations reached."
            return False

        gradient = self.f.gradient(self.current_x)
        new_x = self.current_x - self.step_size * gradient
        new_fx = self.f.objective(new_x)

        if np.abs(new_fx - self.current_fx) < self.obj_tol:
            self.success = True
            self.output_message = "Converged to the objective tolerance."
            return False

        if np.linalg.norm(new_x - self.current_x) < self.param_tol:
            self.success = True
            self.output_message = "Converged to the parameter tolerance."
            return False

        self.current_x = new_x
        self.current_fx = new_fx
        self.history.append((new_x, new_fx))
        self.current_iteration += 1
        return True

    def minimize(self):
        while self.step():
            pass
        return self.current_x, self.current_fx, self.history


class NewtonMethod(LineSearch):
    def __init__(self, f: Function, x0, obj_tol, param_tol, max_iter):
        super().__init__(f, x0, obj_tol, param_tol, max_iter)

    def step(self):
        if self.current_iteration >= self.max_iter:
            return False

        gradient = self.f.gradient(self.current_x)
        hessian = self.f.hessian(self.current_x)
        new_x = self.current_x - np.linalg.solve(hessian, gradient)
        new_fx = self.f.objective(new_x)

        if np.abs(new_fx - self.current_fx) < self.obj_tol:
            self.success = True
            self.output_message = "Converged to the objective tolerance."
            return False

        if np.linalg.norm(new_x - self.current_x) < self.param_tol:
            self.success = True
            self.output_message = "Converged to the parameter tolerance."
            return False

        self.current_x = new_x
        self.current_fx = new_fx
        self.history.append((new_x, new_fx))
        self.current_iteration += 1
        return True, None

    def minimize(self):
        while self.step():
            pass

        print(self.output_message)
        return self.success, self.current_x, self.current_fx, self.history
