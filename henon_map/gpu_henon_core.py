import math
from numba import cuda
import numpy as np
import numba


@cuda.jit(device=True)
def rotation(x, p, angle):
    a = + math.cos(angle) * x + math.sin(angle) * p
    b = - math.sin(angle) * x + math.cos(angle) * p
    return a, b


@cuda.jit(device=True)
def check_boundary(v0, v1, v2, v3, limit):
    return (v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 > limit)


@cuda.jit(device=True)
def polar_to_cartesian(radius, alpha, theta1, theta2):
    x = radius * math.cos(alpha) * math.cos(theta1)
    px = radius * math.cos(alpha) * math.sin(theta1)
    y = radius * math.sin(alpha) * math.cos(theta2)
    py = radius * math.sin(alpha) * math.sin(theta2)
    return x, px, y, py


@cuda.jit(device=True)
def cartesian_to_polar(x, y, px, py):
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) +
                np.power(px, 2) + np.power(py, 2))
    theta1 = np.arctan2(px, x) + np.pi
    theta2 = np.arctan2(py, y) + np.pi
    alpha = np.arctan2(np.sqrt(y * y + py * py),
                       np.sqrt(x * x + px * px)) + np.pi
    return r, alpha, theta1, theta2


@cuda.jit
def dummy_map(step, max_iterations):
    stride = cuda.blockDim.x * cuda.gridDim.x
    for j in range(cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x, step.shape[0], stride):
        step[j] = max_iterations


@cuda.jit
def henon_map(c_alpha, c_theta1, c_theta2, c_dr, step, c_limit, c_max_iterations, omega_x, omega_y):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # const... I hope...
    dr = cuda.shared.array(shape=(1), dtype=numba.float64)
    limit = cuda.shared.array(shape=(1), dtype=numba.float64)
    max_iterations = cuda.shared.array(shape=(1), dtype=numba.int32)
    if i == 0:
        dr[0] = c_dr
        limit[0] = c_limit
        max_iterations[0] = c_max_iterations

    # allocate shared memory
    alpha = cuda.shared.array(shape=(512), dtype=numba.float64)
    theta1 = cuda.shared.array(shape=(512), dtype=numba.float64)
    theta2 = cuda.shared.array(shape=(512), dtype=numba.float64)

    step_local = cuda.shared.array(shape=(512), dtype=numba.int32)

    x = cuda.shared.array(shape=(512), dtype=numba.float64)
    px = cuda.shared.array(shape=(512), dtype=numba.float64)
    y = cuda.shared.array(shape=(512), dtype=numba.float64)
    py = cuda.shared.array(shape=(512), dtype=numba.float64)
    
    temp1 = cuda.shared.array(shape=(512), dtype=numba.float64)
    temp2 = cuda.shared.array(shape=(512), dtype=numba.float64)
    
    cuda.syncthreads()
    
    if j < step.shape[0]:
        # filling
        alpha[i] = c_alpha[j]
        theta1[i] = c_theta1[j]
        theta2[i] = c_theta2[j]
        
        step_local[i] = step[j] + 1
        while True:
            x[i], y[i], px[i], py[i] = polar_to_cartesian(
                dr[0] * step_local[i], alpha[i], theta1[i], theta2[i])
            for k in range(max_iterations[0]):
                temp1[i] = px[i] + x[i] * x[i] - y[i] * y[i]
                temp2[i] = py[i] - 2 * x[i] * y[i]

                x[i], px[i] = rotation(x[i], temp1[i], omega_x[k])
                y[i], py[i] = rotation(y[i], temp2[i], omega_y[k])
                if check_boundary(x[i], y[i], px[i], py[i], limit[0]):
                    step_local[i] -= 1
                    cuda.syncthreads()
                    step[j] = step_local[i]
                    return
            step_local[i] += 1


@cuda.jit
def henon_full_track(x, y, px, py, n_iterations, omega_x, omega_y):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    temp = cuda.shared.array(shape=(512), dtype=numba.float64)
    cuda.syncthreads()

    for k in range(1, n_iterations[j]):
        temp[i] = (px[k - 1][j] 
            + x[k - 1][j] * x[k - 1][j] - y[k - 1][j] * y[k - 1][j])
        x[k][j], px[k][j] = rotation(x[k - 1][j], temp[i], omega_x[k - 1]) 
        
        temp[i] = (py[k - 1][j] 
            - 2 * x[k - 1][j] * y[k - 1][j])
        y[k][j], py[k][j] = rotation(y[k - 1][j], temp[i], omega_y[k - 1])


@cuda.jit
def henon_partial_track(g_x, g_y, g_px, g_py, g_steps, limit, max_iterations, omega_x, omega_y):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    x = cuda.shared.array(shape=(512), dtype=numba.float64)
    y = cuda.shared.array(shape=(512), dtype=numba.float64)
    px = cuda.shared.array(shape=(512), dtype=numba.float64)
    py = cuda.shared.array(shape=(512), dtype=numba.float64)
    temp1 = cuda.shared.array(shape=(512), dtype=numba.float64)
    temp2 = cuda.shared.array(shape=(512), dtype=numba.float64)
    steps = cuda.shard.array(shape=(512), dtype=numba.int32)

    x[i] = g_x[j]
    y[i] = g_x[j]
    px[i] = g_px[j]
    py[i] = g_py[j]
    steps[i] = g_steps[j]

    cuda.syncthreads()

    for k in range(max_iterations):
        temp1[i] = (px[i] + x[i] * x[i] - y[i] * y[i])
        temp2[i] = (py[i] - 2 * x[i] * y[i])

        x[i], px[i] = rotation(x[i], temp1[i], omega_x[k])
        y[i], py[i] = rotation(y[i], temp2[i], omega_y[k])
        if(check_boundary(x[i], px[i], y[i], py[i], limit) or (x[i] == 0.0 and px[i] == 0.0 and y[i] == 0.0 and py[i] == 0.0)):
            x[i] = 0.0
            px[i] = 0.0
            y[i] = 0.0
            py[i] = 0.0
            break
        steps[j] += 1

    cuda.syncthreads()
    g_x[j] = x[i]
    g_y[j] = x[i]
    g_px[j] = px[i]
    g_py[j] = py[i]
    g_steps[j] = steps[i]
