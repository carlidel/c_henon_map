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
    for j in prange(step.size):
        step[j] = max_iterations
    return step


@njit(parallel=True)
def advanced_dummy_map(alpha, theta1, theta2, r):
    step = np.zeros(alpha.size)
    for j in prange(alpha.size):
        step[j] = (
            (alpha[j] 
            + 0.5 * np.sin(theta1[j] * 5) 
            + 0.5 * np.sin(theta2[j] * 3)
            ) * r)
        # step[j] = alpha[j] + theta1[j] + theta2[j]
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


@njit(parallel=True)
def henon_full_track(radius, alpha, theta1, theta2, n_iterations, omega_x, omega_y):

    x = np.empty((n_iterations, radius.size))
    y = np.empty((n_iterations, radius.size))
    px = np.empty((n_iterations, radius.size))
    py = np.empty((n_iterations, radius.size))

    for j in prange(len(radius)):
        x[0][j], y[0][j], px[0][j], py[0][j] = polar_to_cartesian(
            radius[j], alpha[j], theta1[j], theta2[j])
        for k in range(1, n_iterations):
            temp = (px[k - 1][j]
                    + x[k - 1][j] * x[k - 1][j] - y[k - 1][j] * y[k - 1][j])
            x[k][j], px[k][j] = rotation(x[k - 1][j], temp, omega_x[k - 1])

            temp = (py[k - 1][j]
                    - 2 * x[k - 1][j] * y[k - 1][j])
            y[k][j], py[k][j] = rotation(y[k - 1][j], temp, omega_y[k - 1])
            
    return x, y, px, py
