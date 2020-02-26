import math
from numba import jit, njit, prange
import numpy as np
import numba


@njit
def rotation(x, p, angle):
    a = + np.cos(angle) * x + np.sin(angle) * p
    b = - np.sin(angle) * x + np.cos(angle) * p
    return a, b


@njit
def check_boundary(v0, v1, v2, v3, limit):
    return (v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 > limit)


@njit
def polar_to_cartesian(radius, alpha, theta1, theta2):
    x = radius * math.cos(alpha) * math.cos(theta1)
    px = radius * math.cos(alpha) * math.sin(theta1)
    y = radius * math.sin(alpha) * math.cos(theta2)
    py = radius * math.sin(alpha) * math.sin(theta2)
    return x, px, y, py


@njit
def dummy_map(step, max_iterations):
    for j in prange(step.shape[0]):
        step[j] = max_iterations
    return step


@njit(parallel=True)
def henon_map(alpha, theta1, theta2, dr, step, limit, max_iterations, omega_x, omega_y):
    for j in prange(alpha.shape[0]):
        step[j] += 1
        flag = True
        while flag:
            # Obtain cartesian position
            x, y, px, py = polar_to_cartesian(
                dr * step[j], alpha[j], theta1[j], theta2[j])
            for k in range(max_iterations):
                temp1 = px + x * x - y * y
                temp2 = py - 2 * x * y

                x, px = rotation(x, temp1, omega_x[k])
                y, py = rotation(y, temp2, omega_y[k])
                if check_boundary(x, y, px, py, limit):
                    step[j] -= 1
                    flag = False
                    break
            if flag:
                step[j] += 1
    return step