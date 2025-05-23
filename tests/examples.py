import numpy as np
from abc import ABC, abstractmethod


class Function(ABC):
    def __init__(self, hessian_needed=True):
        self.hessian_needed = hessian_needed

    @abstractmethod
    def objective(self, x) -> float:
        pass

    @abstractmethod
    def gradient(self, x) -> np.ndarray:
        pass

    @abstractmethod
    def hessian(self, x) -> np.ndarray:
        pass


class QuadraticFunction(Function):
    def __init__(self, Q, b=np.zeros(2), c=0.0):
        super().__init__(hessian_needed=True)
        self.A = 2 * Q
        self.b = b
        self.c = c

    def objective(self, x) -> float:
        return 0.5 * x.T @ self.A @ x + self.b @ x + self.c

    def gradient(self, x) -> np.ndarray:
        return self.A @ x + self.b

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
        d2fdx0 = 2 + 12 * self.b * x[0] ** 2 - 4 * self.b * x[1]
        d2fdx0x1 = -4 * self.b * x[0]
        d2fdx1 = 2 * self.b
        return np.array([[d2fdx0, d2fdx0x1], [d2fdx0x1, d2fdx1]])


class LinearFunction(Function):
    def __init__(self, a):
        super().__init__(hessian_needed=False)
        self.a = a

    def objective(self, x) -> float:
        return self.a @ x

    def gradient(self, x) -> np.ndarray:
        return self.a

    def hessian(self, x) -> np.ndarray:
        return np.zeros((len(self.a), len(self.a)))


class ExponentialFunction(Function):
    def __init__(self):
        super().__init__(hessian_needed=True)

    def objective(self, x) -> float:
        x1, x2 = x
        return np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) + np.exp(-x1 - 0.1)

    def gradient(self, x) -> np.ndarray:
        x1, x2 = x
        dfdx1 = (
            np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 3 * x2 - 0.1) - np.exp(-x1 - 0.1)
        )
        dfdx2 = 3 * np.exp(x1 + 3 * x2 - 0.1) - 3 * np.exp(x1 - 3 * x2 - 0.1)
        return np.array([dfdx1, dfdx2])

    def hessian(self, x) -> np.ndarray:
        x1, x2 = x
        t1 = np.exp(x1 + 3 * x2 - 0.1)
        t2 = np.exp(x1 - 3 * x2 - 0.1)
        t3 = np.exp(-x1 - 0.1)
        d2fdx1dx1 = t1 + t2 + t3
        d2fdx1dx2 = 3 * t1 - 3 * t2
        d2fdx2dx2 = 9 * t1 + 9 * t2
        return np.array([[d2fdx1dx1, d2fdx1dx2], [d2fdx1dx2, d2fdx2dx2]])
