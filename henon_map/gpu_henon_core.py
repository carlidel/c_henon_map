import math
from numba import cuda
import numpy as np
import numba


@cuda.jit(device=True)
def rotation(x, p, angle, inverse):
    if not inverse:
        a = + math.cos(angle) * x + math.sin(angle) * p
        b = - math.sin(angle) * x + math.cos(angle) * p
    else:
        a = + math.cos(angle) * x - math.sin(angle) * p
        b = + math.sin(angle) * x + math.cos(angle) * p
    return a, b


@cuda.jit(device=True)
def premade_rotation(x, p, sin_a, cos_a, inverse):
    if not inverse:
        a = + cos_a * x + sin_a * p
        b = - sin_a * x + cos_a * p
    else:
        a = + cos_a * x - sin_a * p
        b = + sin_a * x + cos_a * p
    return a, b


@cuda.jit(device=True)
def check_boundary(v0, v1, v2, v3, limit):
    if (math.isnan(v0) or math.isnan(v1) or math.isnan(v2) or math.isnan(v3)):
        return True
    return (v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 > limit)


@cuda.jit(device=True)
def check_boundary2(v02, v12, v22, v32, limit):
    return (v02 + v12 + v22 + v32 > limit)


@cuda.jit(device=True)
def polar_to_cartesian(radius, alpha, theta1, theta2):
    x = radius * math.cos(alpha) * math.cos(theta1)
    px = radius * math.cos(alpha) * math.sin(theta1)
    y = radius * math.sin(alpha) * math.cos(theta2)
    py = radius * math.sin(alpha) * math.sin(theta2)
    return x, px, y, py


@cuda.jit
def dummy_polar_to_cartesian(radius, alpha, theta1, theta2, x, px, y, py):
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if j < radius.shape[0]:
        x[j], px[j], y[j], py[j] = polar_to_cartesian(
            radius[j], alpha[j], theta1[j], theta2[j])


def actual_polar_to_cartesian(radius, alpha, theta1, theta2):
    x = np.zeros(radius.shape)
    px = np.zeros(radius.shape)
    y = np.zeros(radius.shape)
    py = np.zeros(radius.shape)
    d_x = cuda.to_device(np.zeros(radius.shape))
    d_y = cuda.to_device(np.zeros(radius.shape))
    d_px = cuda.to_device(np.zeros(radius.shape))
    d_py = cuda.to_device(np.zeros(radius.shape))

    d_radius = cuda.to_device(np.ascontiguousarray(radius))
    d_alpha = cuda.to_device(np.ascontiguousarray(alpha))
    d_theta1 = cuda.to_device(np.ascontiguousarray(theta1))
    d_theta2 = cuda.to_device(np.ascontiguousarray(theta2))

    dummy_polar_to_cartesian[radius.size//1024 + 1,
                             1024](d_radius, d_alpha, d_theta1, d_theta2, d_x, d_px, d_y, d_py)

    d_x.copy_to_host(x)
    d_px.copy_to_host(px)
    d_y.copy_to_host(y)
    d_py.copy_to_host(py)
    return x, px, y, py


@cuda.jit(device=True)
def cartesian_to_polar(x, px, y, py):
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) +
                np.power(px, 2) + np.power(py, 2))
    theta1 = np.arctan2(px, x) + np.pi
    theta2 = np.arctan2(py, y) + np.pi
    alpha = np.arctan2(np.sqrt(y * y + py * py),
                       np.sqrt(x * x + px * px)) + np.pi
    return r, alpha, theta1, theta2


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
        while j < step.shape[0]:
            x[i], px[i], y[i], py[i] = polar_to_cartesian(
                dr[0] * step_local[i], alpha[i], theta1[i], theta2[i])
            for k in range(max_iterations[0]):
                temp1[i] = px[i] + x[i] * x[i] - y[i] * y[i]
                temp2[i] = py[i] - 2 * x[i] * y[i]

                x[i], px[i] = rotation(
                    x[i], temp1[i], omega_x[k], False)
                y[i], py[i] = rotation(
                    y[i], temp2[i], omega_y[k], False)
                if check_boundary(x[i], px[i], y[i], py[i], limit[0]):
                    step_local[i] -= 1
                    cuda.syncthreads()
                    step[j] = step_local[i]
                    return
            step_local[i] += 1


@cuda.jit
def henon_map_to_the_end(c_x, c_px, c_y, c_py, steps, c_limit, c_max_iterations, omega_x, omega_y, bool_mask):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # const... I hope...
    limit = cuda.shared.array(shape=(1), dtype=numba.float64)
    max_iterations = cuda.shared.array(shape=(1), dtype=numba.int32)
    if i == 0:
        limit[0] = c_limit
        max_iterations[0] = c_max_iterations

    # allocate shared memory
    x = cuda.shared.array(shape=(512), dtype=numba.float64)
    px = cuda.shared.array(shape=(512), dtype=numba.float64)
    y = cuda.shared.array(shape=(512), dtype=numba.float64)
    py = cuda.shared.array(shape=(512), dtype=numba.float64)

    temp1 = cuda.shared.array(shape=(512), dtype=numba.float64)
    temp2 = cuda.shared.array(shape=(512), dtype=numba.float64)

    # begin with the radial optimized loop
    while j < steps.shape[0]:  # Are we still inside the valid loop?

        if steps[j] == 0:  # Are we using a new initial condition?
            # filling the new initial condition
            x[i] = c_x[j]
            px[i] = c_px[j]
            y[i] = c_y[j]
            py[i] = c_py[j]

        # Henon map iteration
        temp1[i] = px[i] + x[i] * x[i] - y[i] * y[i]
        temp2[i] = py[i] - 2 * x[i] * y[i]

        x[i], px[i] = rotation(
            x[i], temp1[i], omega_x[int(steps[j])], False)
        y[i], py[i] = rotation(
            y[i], temp2[i], omega_y[int(steps[j])], False)

        steps[j] += 1

        # Have we lost the particle OR have we hit the limit OR was that useless?
        if (check_boundary(x[i], px[i], y[i], py[i], limit[0]) or steps[j] > max_iterations[0] or (not bool_mask[j])):
            # Remove last step OR fix it
            if bool_mask[j]:
                steps[j] -= 1
            else:
                steps[j] = max_iterations[0]
            # Block skip to next initial condition!
            j += 512 * 10
    return


@cuda.jit
def octo_henon_map_to_the_end(c_x, c_px, c_y, c_py, steps, c_limit, c_max_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, c_mu, bool_mask):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # Let's make it float64...

    limit = cuda.shared.array(shape=(1), dtype=numba.float64)
    max_iterations = cuda.shared.array(shape=(1), dtype=numba.int32)
    mu = cuda.shared.array(shape=(1), dtype=numba.float64)
    if i == 0:
        limit[0] = c_limit
        max_iterations[0] = c_max_iterations
        mu[0] = c_mu
    
    cuda.syncthreads()
    
    # allocate shared memory
    x = cuda.shared.array(shape=(512), dtype=numba.float64)
    px = cuda.shared.array(shape=(512), dtype=numba.float64)
    y = cuda.shared.array(shape=(512), dtype=numba.float64)
    py = cuda.shared.array(shape=(512), dtype=numba.float64)

    x2 = cuda.shared.array(shape=(512), dtype=numba.float64)
    y2 = cuda.shared.array(shape=(512), dtype=numba.float64)

    temp1 = cuda.shared.array(shape=(512), dtype=numba.float64)
    temp2 = cuda.shared.array(shape=(512), dtype=numba.float64)

    # begin with the radial optimized loop
    while j < steps.shape[0]:  # Are we still inside the valid loop?

        if steps[j] == 0:  # Are we using a new initial condition?
            # filling the new initial condition
            x[i] = c_x[j]
            px[i] = c_px[j]
            y[i] = c_y[j]
            py[i] = c_py[j]
            x2[i] = x[i] * x[i]
            y2[i] = y[i] * y[i]

        # Henon map iteration
        temp1[i] = px[i] + x2[i] - y2[i] + mu[0] * (x2[i] * x[i] - 3 * x[i] * y2[i])
        temp2[i] = py[i] - 2 * x[i] * y[i] + mu[0] * (3 * x2[i] * y[i] - y2[i] * y[i])

        x[i], px[i] = premade_rotation(
            x[i], temp1[i], omega_x_sin[int(steps[j])], omega_x_cos[int(steps[j])], False)
        y[i], py[i] = premade_rotation(
            y[i], temp2[i], omega_y_sin[int(steps[j])], omega_y_cos[int(steps[j])], False)

        x2[i] = x[i] * x[i]
        y2[i] = y[i] * y[i]

        steps[j] += 1

        # Have we lost the particle OR have we hit the limit OR was that useless?
        if (check_boundary2(x2[i], px[i] * px[i], y2[i], py[i] * py[i], limit[0]) or steps[j] > max_iterations[0] or (not bool_mask[j])):
            # Remove last step OR fix it
            if bool_mask[j]:
                steps[j] -= 1
            else:
                steps[j] = max_iterations[0]
            # Block skip to next initial condition!
            j += 512 * 10
    return



@cuda.jit
def henon_full_track(x, px, y, py, n_iterations, omega_x, omega_y):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    k = 1    
    temp = cuda.shared.array(shape=(512), dtype=numba.float64)
    
    while j < x.shape[1]:
        temp[i] = (px[k - 1][j] 
            + x[k - 1][j] * x[k - 1][j] - y[k - 1][j] * y[k - 1][j])
        x[k][j], px[k][j] = rotation(
            x[k - 1][j], temp[i], omega_x[k - 1], False)
        temp[i] = (py[k - 1][j] 
            - 2 * x[k - 1][j] * y[k - 1][j])
        y[k][j], py[k][j] = rotation(
            y[k - 1][j], temp[i], omega_y[k - 1], False)
        if(check_boundary(x[k][j], px[k][j], y[k][j], py[k][j], 1.0) or k >= n_iterations[j]):
            x[k][j] = np.nan
            px[k][j] = np.nan
            y[k][j] = np.nan
            py[k][j] = np.nan
            n_iterations[j] = k
            j += 512 * 10
            k = 1
        else:
            k += 1


@cuda.jit
def henon_partial_track(g_x, g_px, g_y, g_py, g_steps, limit, max_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    x = cuda.shared.array(shape=(512), dtype=numba.float64)
    y = cuda.shared.array(shape=(512), dtype=numba.float64)
    px = cuda.shared.array(shape=(512), dtype=numba.float64)
    py = cuda.shared.array(shape=(512), dtype=numba.float64)
    temp1 = cuda.shared.array(shape=(512), dtype=numba.float64)
    temp2 = cuda.shared.array(shape=(512), dtype=numba.float64)
    steps = cuda.shared.array(shape=(512), dtype=numba.int32)

    if(j < g_x.shape[0]):
        x[i] = g_x[j]
        y[i] = g_y[j]
        px[i] = g_px[j]
        py[i] = g_py[j]
        steps[i] = g_steps[j]

        for k in range(max_iterations):
            temp1[i] = (px[i] + x[i] * x[i] - y[i] * y[i])
            temp2[i] = (py[i] - 2 * x[i] * y[i])

            x[i], px[i] = premade_rotation(
                x[i], temp1[i], omega_x_sin[k], omega_x_cos[k], False)
            y[i], py[i] = premade_rotation(
                y[i], temp2[i], omega_y_sin[k], omega_y_cos[k], False)

            if(check_boundary(x[i], px[i], y[i], py[i], limit) or (math.isnan(x[i]) or math.isnan(px[i]) or math.isnan(y[i]) or math.isnan(py[i]))):
                x[i] = math.nan
                px[i] = math.nan
                y[i] = math.nan
                py[i] = math.nan
                break
            steps[i] += 1

        g_x[j] = x[i]
        g_y[j] = y[i]
        g_px[j] = px[i]
        g_py[j] = py[i]
        g_steps[j] = steps[i]


@cuda.jit
def henon_inverse_partial_track(g_x, g_px, g_y, g_py, g_steps, limit, max_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    x = cuda.shared.array(shape=(512), dtype=numba.float64)
    y = cuda.shared.array(shape=(512), dtype=numba.float64)
    px = cuda.shared.array(shape=(512), dtype=numba.float64)
    py = cuda.shared.array(shape=(512), dtype=numba.float64)
    steps = cuda.shared.array(shape=(512), dtype=numba.int32)

    if(j < g_x.shape[0]):
        x[i] = g_x[j]
        y[i] = g_y[j]
        px[i] = g_px[j]
        py[i] = g_py[j]
        steps[i] = g_steps[j]

        for k in range(max_iterations):
            x[i], px[i] = premade_rotation(
                x[i], px[i], omega_x_sin[k], omega_x_cos[k], True)
            y[i], py[i] = premade_rotation(
                y[i], py[i], omega_y_sin[k], omega_y_cos[k], True)

            px[i] = px[i] - (x[i] * x[i] - y[i] * y[i])
            py[i] = py[i] + 2 * x[i] * y[i]

            if(check_boundary(x[i], px[i], y[i], py[i], limit) or (math.isnan(x[i]) or math.isnan(px[i]) or math.isnan(y[i]) or math.isnan(py[i]))):
                x[i] = math.nan
                px[i] = math.nan
                y[i] = math.nan
                py[i] = math.nan
                break
            steps[i] -= 1

        g_x[j] = x[i]
        g_y[j] = y[i]
        g_px[j] = px[i]
        g_py[j] = py[i]
        g_steps[j] = steps[i]


@cuda.jit
def octo_henon_partial_track(g_x, g_px, g_y, g_py, g_steps, limit, max_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, mu):
    i = cuda.threadIdx.x
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    x = cuda.shared.array(shape=(512), dtype=numba.float64)
    y = cuda.shared.array(shape=(512), dtype=numba.float64)
    px = cuda.shared.array(shape=(512), dtype=numba.float64)
    py = cuda.shared.array(shape=(512), dtype=numba.float64)

    x2 = cuda.shared.array(shape=(512), dtype=numba.float64)
    y2 = cuda.shared.array(shape=(512), dtype=numba.float64)

    temp1 = cuda.shared.array(shape=(512), dtype=numba.float64)
    temp2 = cuda.shared.array(shape=(512), dtype=numba.float64)
    steps = cuda.shared.array(shape=(512), dtype=numba.int32)

    if(j < g_x.shape[0]):
        x[i] = g_x[j]
        y[i] = g_y[j]
        px[i] = g_px[j]
        py[i] = g_py[j]
        steps[i] = g_steps[j]
        
        x2[i] = x[i] * x[i]
        y2[i] = y[i] * y[i]

        for k in range(max_iterations):
            temp1[i] = (px[i] + x2[i] - y2[i] + mu *
                        (x2[i] * x[i] - 3 * x[i] * y2[i]))
            temp2[i] = (py[i] - 2 * x[i] * y[i] + mu *
                        (3 * x2[i] * y[i] - y2[i] * y[i]))

            x[i], px[i] = premade_rotation(
                x[i], temp1[i], omega_x_sin[k], omega_x_cos[k], False)
            y[i], py[i] = premade_rotation(
                y[i], temp2[i], omega_y_sin[k], omega_y_cos[k], False)

            if(check_boundary(x[i], px[i], y[i], py[i], limit) or (math.isnan(x[i]) or math.isnan(px[i]) or math.isnan(y[i]) or math.isnan(py[i]))):
                x[i] = math.nan
                px[i] = math.nan
                y[i] = math.nan
                py[i] = math.nan
                break
            steps[i] += 1
            x2[i] = x[i] * x[i]
            y2[i] = y[i] * y[i]

        g_x[j] = x[i]
        g_y[j] = y[i]
        g_px[j] = px[i]
        g_py[j] = py[i]
        g_steps[j] = steps[i]
