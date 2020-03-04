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
    return (v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3) * 0.5 > limit


@njit
def polar_to_cartesian(radius, alpha, theta1, theta2):
    x = radius * np.cos(alpha) * np.cos(theta1)
    px = radius * np.cos(alpha) * np.sin(theta1)
    y = radius * np.sin(alpha) * np.cos(theta2)
    py = radius * np.sin(alpha) * np.sin(theta2)
    return x, px, y, py


@njit
def cartesian_to_polar(x, y, px, py):
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) +
                np.power(px, 2) + np.power(py, 2))
    theta1 = np.arctan2(px, x)
    theta2 = np.arctan2(py, y)
    alpha = np.arctan2(np.sqrt(y * y + py * py),
                       np.sqrt(x * x + px * px))
    return r, alpha, theta1, theta2


@njit
def dummy_map(step, max_iterations):
    for j in prange(step.size):
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


@njit(parallel=True)
def henon_map_2D(x, p, n_iters, limit, max_iterations, omega):
    for j in prange(x.size):
        for k in range(max_iterations):
            temp = p[j] + x[j] * x[j]
            x[j], p[j] = rotation(x[j], temp, omega[k])
            if ((x[j] * x[j] + p[j] * p[j]) * 0.5 > limit 
                or (x[j] == 0 and p[j] == 0)):
                x[j] = 0.0
                p[j] = 0.0
                flag = False
                break
            n_iters[j] += 1
    return x, p, n_iters


@njit(parallel=True)
def henon_full_track(x, y, px, py, n_iterations, omega_x, omega_y):
    for j in prange(x.shape[2]):
        for k in range(1, n_iterations[j]):
            temp = (px[k - 1][j]
                    + x[k - 1][j] * x[k - 1][j] - y[k - 1][j] * y[k - 1][j])
            x[k][j], px[k][j] = rotation(x[k - 1][j], temp, omega_x[k - 1])

            temp = (py[k - 1][j]
                    - 2 * x[k - 1][j] * y[k - 1][j])
            y[k][j], py[k][j] = rotation(y[k - 1][j], temp, omega_y[k - 1])
            
    return x, y, px, py


@njit(parallel=True)
def henon_partial_track(x, y, px, py, steps, limit, max_iterations, omega_x, omega_y):
    for j in prange(len(x)):
        for k in range(max_iterations):
            temp1 = (px[j] + x[j] * x[j] - y[j] * y[j])
            temp2 = (py[j] - 2 * x[j] * y[j])

            x[j], px[j] = rotation(x[j], temp1, omega_x[k])
            y[j], py[j] = rotation(y[j], temp2, omega_y[k])
            if(check_boundary(x[j], px[j], y[j], py[j], limit) or (x[j] == 0.0 and px[j] == 0.0 and y[j] == 0.0 and py[j] == 0.0)):
                x[j] = 0.0
                px[j] = 0.0
                y[j] = 0.0
                py[j] = 0.0
                break
            steps[j] += 1
    return x, y, px, py, steps
