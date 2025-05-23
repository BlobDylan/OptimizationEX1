import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.unconstrained_min import GradientDescent, NewtonMethod
from examples import (
    QuadraticFunction,
    RosenbrockFunction,
    LinearFunction,
    ExponentialFunction,
)
from src import utils


class TestUnconstrainedMin(unittest.TestCase):
    def setUp(self):
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 100
        self.rosenbrock_max_iter = 10000

    def run_example(self, func, x0, max_iter, example_name):
        gd = GradientDescent(func, x0, self.obj_tol, self.param_tol, max_iter)
        x_gd, f_gd, success_gd, hist_gd = gd.minimize()
        nt = NewtonMethod(func, x0, self.obj_tol, self.param_tol, max_iter)
        x_nt, f_nt, success_nt, hist_nt = nt.minimize()
        utils.plot_contour(
            func.objective,
            [-2, 2],
            [-2, 2],
            {"GD": hist_gd, "Newton": hist_nt},
            f"{example_name} Contour",
        )
        plt.savefig(f"{example_name}_contour.png")
        plt.close()
        utils.plot_function_values(
            {"GD": hist_gd, "Newton": hist_nt}, f"{example_name} Function Values"
        )
        plt.savefig(f"{example_name}_fvals.png")
        plt.close()
        print(f"\n{example_name} - GD: Success={success_gd}, x={x_gd}, f={f_gd}")
        print(f"{example_name} - Newton: Success={success_nt}, x={x_nt}, f={f_nt}")

    def test_quadratic1(self):
        Q = np.eye(2)
        func = QuadraticFunction(Q)
        self.run_example(func, np.array([1.0, 1.0]), self.max_iter, "Quadratic1")

    def test_quadratic2(self):
        Q = np.diag([1, 100])
        func = QuadraticFunction(Q)
        self.run_example(func, np.array([1.0, 1.0]), self.max_iter, "Quadratic2")

    def test_quadratic3(self):
        rotation = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
        Q = rotation.T @ np.diag([100, 1]) @ rotation
        func = QuadraticFunction(Q)
        self.run_example(func, np.array([1.0, 1.0]), self.max_iter, "Quadratic3")

    def test_rosenbrock(self):
        func = RosenbrockFunction()
        self.run_example(
            func, np.array([-1.0, 2.0]), self.rosenbrock_max_iter, "Rosenbrock"
        )

    def test_linear(self):
        a = np.array([2.0, 3.0])
        func = LinearFunction(a)
        self.run_example(func, np.array([1.0, 1.0]), self.max_iter, "Linear")

    def test_exponential(self):
        func = ExponentialFunction()
        self.run_example(func, np.array([1.0, 1.0]), self.max_iter, "Exponential")


if __name__ == "__main__":
    unittest.main()
