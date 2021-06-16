import numpy as np


def curvature(x, y):
    dx = np.gradient(x)   #x'
    dy = np.gradient(y)   #y'

    d2x = np.gradient(dx)    #x''
    d2y = np.gradient(dy)    #y''

    return (dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5