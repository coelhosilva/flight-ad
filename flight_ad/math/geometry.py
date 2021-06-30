import numpy as np


def curvature(x, y):
    dx = np.gradient(x)
    dy = np.gradient(y)

    d2x = np.gradient(dx)
    d2y = np.gradient(dy)

    return (dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5
