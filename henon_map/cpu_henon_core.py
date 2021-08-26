import math
from numba import jit, njit, prange
import numpy as np
import numba


@njit
def rotation(x, p, angle, inverse):
    if not inverse:
        a = + np.cos(angle) * x + np.sin(angle) * p
        b = - np.sin(angle) * x + np.cos(angle) * p
    else:
        a = + np.cos(angle) * x - np.sin(angle) * p
        b = + np.sin(angle) * x + np.cos(angle) * p
    return a, b


@njit
def premade_rotation(x, p, sin_a, cos_a, inverse):
    if not inverse:
        a = + cos_a * x + sin_a * p
        b = - sin_a * x + cos_a * p
    else:
        a = + cos_a * x - sin_a * p
        b = + sin_a * x + cos_a * p
    return a, b


@njit
def check_boundary(v0, v1, v2, v3, limit):
    if (math.isnan(v0) or math.isnan(v1) or math.isnan(v2) or math.isnan(v3)):
        return True
    return (v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3) * 0.5 > limit


@njit
def polar_to_cartesian(radius, alpha, theta1, theta2):
    x = radius * np.cos(alpha) * np.cos(theta1)
    px = radius * np.cos(alpha) * np.sin(theta1)
    y = radius * np.sin(alpha) * np.cos(theta2)
    py = radius * np.sin(alpha) * np.sin(theta2)
    return x, px, y, py


@njit
def cartesian_to_polar(x, px, y, py):
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) +
                np.power(px, 2) + np.power(py, 2))
    theta1 = np.arctan2(px, x)
    theta2 = np.arctan2(py, y)
    alpha = np.arctan2(np.sqrt(y * y + py * py),
                       np.sqrt(x * x + px * px))
    return r, alpha, theta1, theta2


@njit(parallel=True)
def henon_partial_track(x, px, y, py, steps, limit, max_iterations, sin_omega_x, cos_omega_x, sin_omega_y, cos_omega_y):
    for j in prange(len(x)):
        for k in range(max_iterations):
            temp1 = (px[j] + x[j] * x[j] - y[j] * y[j])
            temp2 = (py[j] - 2 * x[j] * y[j])

            x[j], px[j] = premade_rotation(
                x[j], temp1, sin_omega_x[k], cos_omega_x[k], inverse=False)
            y[j], py[j] = premade_rotation(
                y[j], temp2, sin_omega_y[k], cos_omega_y[k], inverse=False)
            if((np.isnan(x[j]) or np.isnan(px[j]) or np.isnan(y[j]) or np.isnan(py[j])) or check_boundary(x[j], px[j], y[j], py[j], limit)):
                x[j] = np.nan
                px[j] = np.nan
                y[j] = np.nan
                py[j] = np.nan
                break
            steps[j] += 1
    return x, px, y, py, steps


@njit(parallel=True)
def henon_inverse_partial_track(x, px, y, py, steps, limit, max_iterations, sin_omega_x, cos_omega_x, sin_omega_y, cos_omega_y):
    for j in prange(len(x)):
        for k in range(max_iterations):
            x[j], px[j] = premade_rotation(
                x[j], px[j], sin_omega_x[k], cos_omega_x[k], inverse=True)
            y[j], py[j] = premade_rotation(
                y[j], py[j], sin_omega_y[k], cos_omega_y[k], inverse=True)

            px[j] = px[j] - (x[j] * x[j] - y[j] * y[j])
            py[j] = py[j] + 2 * x[j] * y[j]

            if((np.isnan(x[j]) or np.isnan(px[j]) or np.isnan(y[j]) or np.isnan(py[j])) or check_boundary(x[j], px[j], y[j], py[j], limit)):
                x[j] = np.nan
                px[j] = np.nan
                y[j] = np.nan
                py[j] = np.nan
                break
            steps[j] -= 1
    return x, px, y, py, steps


@njit(parallel=True)
def octo_henon_partial_track(x, px, y, py, steps, limit, max_iterations, sin_omega_x, cos_omega_x, sin_omega_y, cos_omega_y, mu):
    for j in prange(len(x)):
        for k in range(max_iterations):
            temp1 = (
                px[j]
                + x[j] * x[j]
                - y[j] * y[j]
                + mu * (
                    x[j] * x[j] * x[j]
                    - 3 * x[j] * y[j] * y[j]))
            temp2 = (
                py[j]
                - 2 * x[j] * y[j]
                + mu * (
                    3 * x[j] * x[j] * y[j]
                    - y[j] * y[j] * y[j]))
            x[j], px[j] = premade_rotation(
                x[j], temp1, sin_omega_x[k], cos_omega_x[k], inverse=False)
            y[j], py[j] = premade_rotation(
                y[j], temp2, sin_omega_y[k], cos_omega_y[k], inverse=False)
            if((np.isnan(x[j]) or np.isnan(px[j]) or np.isnan(y[j]) or np.isnan(py[j])) or check_boundary(x[j], px[j], y[j], py[j], limit)):
                x[j] = np.nan
                px[j] = np.nan
                y[j] = np.nan
                py[j] = np.nan
                break
            steps[j] += 1
    return x, px, y, py, steps


@njit(parallel=True)
def octo_henon_inverse_partial_track(x, px, y, py, steps, limit, max_iterations, sin_omega_x, cos_omega_x, sin_omega_y, cos_omega_y, mu):
    for j in prange(len(x)):
        for k in range(max_iterations):
            x[j], px[j] = premade_rotation(
                x[j], px[j], sin_omega_x[k], cos_omega_x[k], inverse=False)
            y[j], py[j] = premade_rotation(
                y[j], py[j], sin_omega_y[k], cos_omega_y[k], inverse=False)

            px[j] = (
                px[j]
                - x[j] * x[j]
                + y[j] * y[j]
                - mu * (
                    x[j] * x[j] * x[j]
                    - 3 * x[j] * y[j] * y[j]))
            py[j] = (
                py[j]
                + 2 * x[j] * y[j]
                - mu * (
                    3 * x[j] * x[j] * y[j]
                    - y[j] * y[j] * y[j]))
            
            if((np.isnan(x[j]) or np.isnan(px[j]) or np.isnan(y[j]) or np.isnan(py[j])) or check_boundary(x[j], px[j], y[j], py[j], limit)):
                x[j] = np.nan
                px[j] = np.nan
                y[j] = np.nan
                py[j] = np.nan
                break
            steps[j] += 1
    return x, px, y, py, steps


@njit(parallel=True)
def henon_partial_track_with_kick(x, px, y, py, steps, limit, max_iterations, sin_omega_x, cos_omega_x, sin_omega_y, cos_omega_y, kick_module, kick_sigma):
    for j in prange(len(x)):
        for k in range(max_iterations):
            temp1 = (px[j] + x[j] * x[j] - y[j] * y[j])
            temp2 = (py[j] - 2 * x[j] * y[j])

            x[j], px[j] = premade_rotation(
                x[j], temp1, sin_omega_x[k], cos_omega_x[k], inverse=False)
            y[j], py[j] = premade_rotation(
                y[j], temp2, sin_omega_y[k], cos_omega_y[k], inverse=False)

            t1 = np.random.uniform(-1, 1)
            t2 = np.random.uniform(-1, 1)
            t3 = np.random.uniform(-1, 1)
            t4 = np.random.uniform(-1, 1)
            while t1 ** 2 + t2 ** 2 >= 1:
                t1 = np.random.uniform(-1, 1)
                t2 = np.random.uniform(-1, 1)
            while t3 ** 2 + t4 ** 2 >= 1:
                t3 = np.random.uniform(-1, 1)
                t4 = np.random.uniform(-1, 1)
            kick = np.random.normal(kick_module, kick_sigma)
            t = (1 - t1 ** 2 - t2 ** 2) / (t3 ** 2 + t4 ** 2)
            x[j] += kick * t1
            px[j] += kick * t2
            y[j] += kick * t3 * t
            py[j] += kick * t4 * t

            if((np.isnan(x[j]) or np.isnan(px[j]) or np.isnan(y[j]) or np.isnan(py[j])) or check_boundary(x[j], px[j], y[j], py[j], limit)):
                x[j] = np.nan
                px[j] = np.nan
                y[j] = np.nan
                py[j] = np.nan
                break
            steps[j] += 1
    return x, px, y, py, steps


@njit(parallel=True)
def henon_inverse_partial_track_with_kick(x, px, y, py, steps, limit, max_iterations, sin_omega_x, cos_omega_x, sin_omega_y, cos_omega_y, kick_module, kick_sigma):
    for j in prange(len(x)):
        for k in range(max_iterations):
            x[j], px[j] = premade_rotation(
                x[j], px[j], sin_omega_x[k], cos_omega_x[k], inverse=True)
            y[j], py[j] = premade_rotation(
                y[j], py[j], sin_omega_y[k], cos_omega_y[k], inverse=True)

            px[j] = px[j] - (x[j] * x[j] - y[j] * y[j])
            py[j] = py[j] + 2 * x[j] * y[j]

            t1 = np.random.uniform(-1, 1)
            t2 = np.random.uniform(-1, 1)
            t3 = np.random.uniform(-1, 1)
            t4 = np.random.uniform(-1, 1)
            while t1 ** 2 + t2 ** 2 >= 1:
                t1 = np.random.uniform(-1, 1)
                t2 = np.random.uniform(-1, 1)
            while t3 ** 2 + t4 ** 2 >= 1:
                t3 = np.random.uniform(-1, 1)
                t4 = np.random.uniform(-1, 1)
            kick = np.random.normal(kick_module, kick_sigma)
            t = (1 - t1 ** 2 - t2 ** 2) / (t3 ** 2 + t4 ** 2)
            x[j] += kick * t1
            px[j] += kick * t2
            y[j] += kick * t3 * t
            py[j] += kick * t4 * t

            if((np.isnan(x[j]) or np.isnan(px[j]) or np.isnan(y[j]) or np.isnan(py[j])) or check_boundary(x[j], px[j], y[j], py[j], limit)):
                x[j] = np.nan
                px[j] = np.nan
                y[j] = np.nan
                py[j] = np.nan
                break
            steps[j] -= 1
    return x, px, y, py, steps


@njit(parallel=True)
def octo_henon_partial_track_with_kick(x, px, y, py, steps, limit, max_iterations, sin_omega_x, cos_omega_x, sin_omega_y, cos_omega_y, mu, kick_module, kick_sigma):
    for j in prange(len(x)):
        for k in range(max_iterations):
            temp1 = (
                px[j]
                + x[j] * x[j]
                - y[j] * y[j]
                + mu * (
                    x[j] * x[j] * x[j]
                    - 3 * x[j] * y[j] * y[j]))
            temp2 = (
                py[j]
                - 2 * x[j] * y[j]
                + mu * (
                    3 * x[j] * x[j] * y[j]
                    - y[j] * y[j] * y[j]))
            x[j], px[j] = premade_rotation(
                x[j], temp1, sin_omega_x[k], cos_omega_x[k], inverse=False)
            y[j], py[j] = premade_rotation(
                y[j], temp2, sin_omega_y[k], cos_omega_y[k], inverse=False)

            t1 = np.random.uniform(-1, 1)
            t2 = np.random.uniform(-1, 1)
            t3 = np.random.uniform(-1, 1)
            t4 = np.random.uniform(-1, 1)
            while t1 ** 2 + t2 ** 2 >= 1:
                t1 = np.random.uniform(-1, 1)
                t2 = np.random.uniform(-1, 1)
            while t3 ** 2 + t4 ** 2 >= 1:
                t3 = np.random.uniform(-1, 1)
                t4 = np.random.uniform(-1, 1)
            kick = np.random.normal(kick_module, kick_sigma)
            t = (1 - t1 ** 2 - t2 ** 2) / (t3 ** 2 + t4 ** 2)
            x[j] += kick * t1
            px[j] += kick * t2
            y[j] += kick * t3 * t
            py[j] += kick * t4 * t

            if((np.isnan(x[j]) or np.isnan(px[j]) or np.isnan(y[j]) or np.isnan(py[j])) or check_boundary(x[j], px[j], y[j], py[j], limit)):
                x[j] = np.nan
                px[j] = np.nan
                y[j] = np.nan
                py[j] = np.nan
                break
            steps[j] += 1
    return x, px, y, py, steps


@njit(parallel=True)
def octo_henon_inverse_partial_track_with_kick(x, px, y, py, steps, limit, max_iterations, sin_omega_x, cos_omega_x, sin_omega_y, cos_omega_y, mu, kick_module, kick_sigma):
    for j in prange(len(x)):
        for k in range(max_iterations):
            x[j], px[j] = premade_rotation(
                x[j], px[j], sin_omega_x[k], cos_omega_x[k], inverse=False)
            y[j], py[j] = premade_rotation(
                y[j], py[j], sin_omega_y[k], cos_omega_y[k], inverse=False)

            px[j] = (
                px[j]
                - x[j] * x[j]
                + y[j] * y[j]
                - mu * (
                    x[j] * x[j] * x[j]
                    - 3 * x[j] * y[j] * y[j]))
            py[j] = (
                py[j]
                + 2 * x[j] * y[j]
                - mu * (
                    3 * x[j] * x[j] * y[j]
                    - y[j] * y[j] * y[j]))

            t1 = np.random.uniform(-1, 1)
            t2 = np.random.uniform(-1, 1)
            t3 = np.random.uniform(-1, 1)
            t4 = np.random.uniform(-1, 1)
            while t1 ** 2 + t2 ** 2 >= 1:
                t1 = np.random.uniform(-1, 1)
                t2 = np.random.uniform(-1, 1)
            while t3 ** 2 + t4 ** 2 >= 1:
                t3 = np.random.uniform(-1, 1)
                t4 = np.random.uniform(-1, 1)
            kick = np.random.normal(kick_module, kick_sigma)
            t = (1 - t1 ** 2 - t2 ** 2) / (t3 ** 2 + t4 ** 2)
            x[j] += kick * t1
            px[j] += kick * t2
            y[j] += kick * t3 * t
            py[j] += kick * t4 * t

            if((np.isnan(x[j]) or np.isnan(px[j]) or np.isnan(y[j]) or np.isnan(py[j])) or check_boundary(x[j], px[j], y[j], py[j], limit)):
                x[j] = np.nan
                px[j] = np.nan
                y[j] = np.nan
                py[j] = np.nan
                break
            steps[j] += 1
    return x, px, y, py, steps
