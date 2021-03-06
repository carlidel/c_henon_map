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
def henon_map(alpha, theta1, theta2, dr, step, limit, max_iterations, omega_x, omega_y):
    for j in prange(alpha.size):
        step[j] += 1
        flag = True
        while True:
            # Obtain cartesian position
            x, px, y, py = polar_to_cartesian(
                dr * step[j], alpha[j], theta1[j], theta2[j])
           
            for k in range(max_iterations):
                temp1 = px + x * x - y * y
                temp2 = py - 2 * x * y

                x, px = rotation(x, temp1, omega_x[k])
                y, py = rotation(y, temp2, omega_y[k])
                if check_boundary(x, px, y, py, limit):
                    step[j] -= 1
                    flag = False
                    break
            if flag:
                step[j] += 1
                continue
            break
    return step


@njit(parallel=True)
def henon_map_to_the_end(c_x, c_px, c_y, c_py, limit, max_iterations, omega_x, omega_y, bool_mask):
    steps = np.zeros_like(c_x)
    for j in prange(len(steps)):
        if bool_mask[j]:
            i = int(steps[j])
            x = c_x[j]
            px = c_px[j]
            y = c_y[j]
            py = c_py[j]

            while not (check_boundary(x, px, y, py, limit) or steps[j] > max_iterations):
                temp1 = px + x * x - y * y
                temp2 = py - 2 * x * y
                
                x, px = rotation(x, temp1, omega_x[i])
                y, py = rotation(y, temp2, omega_y[i])
                
                i += 1
                steps[j] += 1
            i -= 1
            steps[j] -= 1
        else:
            steps[j] = max_iterations
    return steps


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
def henon_full_track(x, px, y, py, n_iterations, omega_x, omega_y):
    for j in prange(x.shape[1]):
        for k in range(1, n_iterations[j]):
            temp = (px[k - 1][j]
                    + x[k - 1][j] * x[k - 1][j] - y[k - 1][j] * y[k - 1][j])
            x[k][j], px[k][j] = rotation(x[k - 1][j], temp, omega_x[k - 1])

            temp = (py[k - 1][j]
                    - 2 * x[k - 1][j] * y[k - 1][j])
            y[k][j], py[k][j] = rotation(y[k - 1][j], temp, omega_y[k - 1])

            if (check_boundary(x[k][j], px[k][j], y[k][j], py[k][j], 1.0)):
                x[k][j] = np.nan
                px[k][j] = np.nan
                y[k][j] = np.nan
                py[k][j] = np.nan
                n_iterations[j] = k - 1
                break
            
    return x, px, y, py


@njit(parallel=True)
def accumulate_and_return(r, alpha, th1, th2, n_sectors):
    tmp_1 = ((th1 + np.pi) / (np.pi * 2)) * n_sectors
    tmp_2 = ((th2 + np.pi) / (np.pi * 2)) * n_sectors
    
    i_1 = np.empty(tmp_1.shape, dtype=np.int32)
    i_2 = np.empty(tmp_2.shape, dtype=np.int32)

    for i in prange(i_1.shape[0]):
        for j in range(i_1.shape[1]):
            i_1[i, j] = int(tmp_1[i, j])
            i_2[i, j] = int(tmp_2[i, j])

    result = np.empty(r.shape[1])
    matrices = np.empty((r.shape[1], n_sectors, n_sectors))
    count = np.zeros((r.shape[1], n_sectors, n_sectors), dtype=np.int32)

    for j in prange(r.shape[1]):
        matrix = np.zeros((n_sectors, n_sectors)) * np.nan

        for k in range(r.shape[0]):
            if count[j, i_1[k, j], i_2[k, j]] == 0:
                matrix[i_1[k, j], i_2[k, j]] = r[k, j]
            else:
                matrix[i_1[k, j], i_2[k, j]] = (
                    (matrix[i_1[k, j], i_2[k, j]] * count[j, i_1[k, j],
                                                          i_2[k, j]] + r[k, j]) / (count[j, i_1[k, j], i_2[k, j]] + 1)
                )
            count[j, i_1[k, j], i_2[k, j]] += 1
        
        result[j] = np.nanmean(np.power(matrix, 4))
        matrices[j,:,:] = matrix
    
    return count, matrices, result


def recursive_accumulation(count, matrices):
    n_sectors = count.shape[1]
    c = []
    m = []
    r = []
    count = count.copy()
    matrices = matrices.copy()
    validity = []
    c.append(count.copy())
    m.append(matrices.copy())
    r.append(np.nanmean(np.power(matrices, 4), axis=(1,2)))
    validity.append(np.logical_not(np.any(np.isnan(matrices), axis=(1, 2))))
    while n_sectors >= 2 and n_sectors % 2 == 0:
        matrices *= count
        count = np.nansum(count.reshape(
            (count.shape[0], n_sectors//2, 2, n_sectors//2, 2)), axis=(2, 4))
        matrices = np.nansum(matrices.reshape(
            (matrices.shape[0], n_sectors//2, 2, n_sectors//2, 2)), axis=(2, 4)) / count
        result = np.nanmean(np.power(matrices, 4), axis=(1,2))
        c.append(count.copy())
        m.append(matrices.copy())
        r.append(result.copy())
        validity.append(np.logical_not(np.any(np.isnan(matrices), axis=(1,2))))
        n_sectors = n_sectors // 2
    return c, m, r, np.asarray(validity, dtype=np.bool)


@njit(parallel=True)
def henon_partial_track(x, px, y, py, steps, limit, max_iterations, omega_x, omega_y):
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
    return x, px, y, py, steps
