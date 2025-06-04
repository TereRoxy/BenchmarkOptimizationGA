import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define Ackley function
def ackley(x, y):
    """
    Ackley function in 2D.
    Domain: [-10, 10] x [-10, 10]
    Global minimum: f(0, 0) = 0
    a = 20, b = 0.2, c = 2 * pi (as recommended in the bibliographic reference)
    """
    term1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    return term1 + term2 + 20 + np.e

# Define Rastrigin function
def rastrigin(x, y):
    """
    Rastrigin function in 2D.
    Domain: [-5, 5] x [-5, 5]
    Global minimum: f(0, 0) = 0
    """
    return 20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

# Function to create contour and surface plots
def plot_function(func, name, x_range, y_range, save_prefix):
    """
    Generate 2D contour and 3D surface plots for a given function.
    Parameters:
        func: Function to plot (e.g., ackley, rastrigin)
        name: Name of the function (str)
        x_range: Tuple of (min, max) for x-axis
        y_range: Tuple of (min, max) for y-axis
        save_prefix: Prefix for saved plot filenames
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    # 2D Contour Plot
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour)
    plt.title(f'{name} Function - Contour Plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'{save_prefix}_contour.png', bbox_inches='tight')
    plt.close()

    # 3D Surface Plot
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_title(f'{name} Function - Surface Plot')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.savefig(f'{save_prefix}_surface.png', bbox_inches='tight')
    plt.close()