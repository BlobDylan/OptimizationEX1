import numpy as np
from abc import ABC, abstractmethod


class Function:
    def __init__(self, hessian_needed=True):
        self.hessian_needed = hessian_needed

    @abstractmethod
    def objective(self, x) -> float:
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def gradient(self, x) -> np.ndarray:
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def hessian(self, x) -> np.ndarray:
        if not self.hessian_needed:
            raise NotImplementedError("Hessian is not needed for this function.")
        raise NotImplementedError("Subclasses should implement this!")


class QuadraticFunction(Function):
    def __init__(self, A: np.ndarray, b: np.ndarray, c: float = 0.0):
        super().__init__(hessian_needed=True)
        self.A = A
        self.b = b
        self.c = c

    def objective(self, x) -> float:
        return 0.5 * np.dot(x.T, np.dot(self.A, x)) + np.dot(self.b, x) + self.c

    def gradient(self, x) -> np.ndarray:
        return np.dot(self.A, x) + self.b

    def hessian(self, x) -> np.ndarray:
        return self.A


class RosenbrockFunction(Function):
    def __init__(self, a=1, b=100):
        super().__init__(hessian_needed=True)
        self.a = a
        self.b = b

    def objective(self, x) -> float:
        return (self.a - x[0]) ** 2 + self.b * (x[1] - x[0] ** 2) ** 2

    def gradient(self, x) -> np.ndarray:
        dfdx0 = -2 * (self.a - x[0]) - 4 * self.b * x[0] * (x[1] - x[0] ** 2)
        dfdx1 = 2 * self.b * (x[1] - x[0] ** 2)
        return np.array([dfdx0, dfdx1])

    def hessian(self, x) -> np.ndarray:
        d2fdx02 = 2 - 4 * self.b * x[1] + 12 * self.b * x[0] ** 2
        d2fdx01 = -4 * self.b * x[0]
        d2fdx11 = 2 * self.b
        return np.array([[d2fdx02, d2fdx01], [d2fdx01, d2fdx11]])


class LinearFunction(Function):
    def __init__(self, A: np.ndarray, b: np.ndarray):
        super().__init__(hessian_needed=False)
        self.A = A
        self.b = b

    def objective(self, x) -> float:
        return np.dot(self.A, x) + self.b

    def gradient(self, x) -> np.ndarray:
        return self.A

    def hessian(self, x) -> np.ndarray:
        raise NotImplementedError("Hessian is not needed for this function.")
