import unittest
import numpy as np
from examples import Function, QuadraticFunction, RosenbrockFunction, LinearFunction
from src.unconstrained_min import LineSearch, GradientDescent


class TestUnconstrainedMin(unittest.TestCase):
    def setUpQuadraticFunctions(self):
        self.quadratic_function1 = QuadraticFunction(
            np.array([[1, 0], [0, 1]]), np.array([0, 0]), 0
        )
        self.quadratic_function2 = QuadraticFunction(
            np.array([[1, 0], [0, 100]]), np.array([0, 0]), 0
        )

    def test_quadratic_functions(self):
        self.setUpQuadraticFunctions()
        x0 = np.array([1, 1])
        obj_tol = 1e-12
        param_tol = 1e-8
        max_iter = 100
        step_size = 0.01

        for func in [self.quadratic_function1, self.quadratic_function2]:
            optimizer = GradientDescent(
                func, x0, obj_tol, param_tol, max_iter, step_size
            )
            optimizer.minimize()

            self.assertTrue(optimizer.success)
            self.assertLessEqual(np.abs(optimizer.current_fx), obj_tol)
            self.assertLessEqual(
                np.linalg.norm(optimizer.current_x - np.zeros_like(x0)), param_tol
            )


if __name__ == "__main__":
    unittest.main()
