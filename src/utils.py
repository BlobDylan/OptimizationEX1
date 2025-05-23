import numpy as np
import matplotlib.pyplot as plt


def plot_contour(func, x_lim, y_lim, paths=None, title="Contour Plot"):
    x = np.linspace(x_lim[0], x_lim[1], 400)
    y = np.linspace(y_lim[0], y_lim[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    min_Z = np.min(Z)
    max_Z = np.max(Z)
    levels = np.logspace(np.log10(min_Z + 1e-6), np.log10(max_Z), 30)

    plt.contour(X, Y, Z, levels=levels, cmap="viridis")
    if paths:
        for method, history in paths.items():
            x_vals = [p[0][0] for p in history]
            y_vals = [p[0][1] for p in history]
            plt.plot(x_vals, y_vals, label=method, marker="o", markersize=3)

    plt.title(title)
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")


def plot_function_values(histories, title="Function Value vs. Iteration"):
    plt.figure()
    for method, history in histories.items():
        f_vals = [h[1] for h in history]
        plt.plot(f_vals, label=method)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Function Value")
    plt.title(title)
    plt.legend()
