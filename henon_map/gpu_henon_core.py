import math
from numba import cuda
import numpy as np
import numba
from numba.cuda.random import xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32


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
def henon_partial_track_with_kick(g_x, g_px, g_y, g_py, g_steps, limit, max_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, rng_states, kick_module, kick_sigma):
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

            # 4d sphere sampling
            t1 = xoroshiro128p_uniform_float32(rng_states, j) * 2 - 1
            t2 = (xoroshiro128p_uniform_float32(
                rng_states, j) * 2 - 1) * math.sqrt(1 - t1)
            t3 = xoroshiro128p_uniform_float32(rng_states, j) * 2 - 1
            t4 = (xoroshiro128p_uniform_float32(
                rng_states, j) * 2 - 1) * math.sqrt(1 - t3)

            # kick module computing
            if kick_sigma == 0:
                kick = kick_module
            else:
                kick = xoroshiro128p_normal_float32(rng_states, j) * kick_sigma + kick_module
            # application
            t = (1 - t1 ** 2 - t2 ** 2) / (t3 ** 2 + t4 ** 2)
            x[i] += kick * t1
            px[i] += kick * t2
            y[i] += kick * t3 * t
            py[i] += kick * t4 * t

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
def henon_inverse_partial_track_with_kick(g_x, g_px, g_y, g_py, g_steps, limit, max_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, rng_states, kick_module, kick_sigma):
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

            # 4d sphere sampling
            t1 = xoroshiro128p_uniform_float32(rng_states, j) * 2 - 1
            t2 = (xoroshiro128p_uniform_float32(
                rng_states, j) * 2 - 1) * math.sqrt(1 - t1)
            t3 = xoroshiro128p_uniform_float32(rng_states, j) * 2 - 1
            t4 = (xoroshiro128p_uniform_float32(
                rng_states, j) * 2 - 1) * math.sqrt(1 - t3)

            # kick module computing
            if kick_sigma == 0:
                kick = kick_module
            else:
                kick = xoroshiro128p_normal_float32(
                    rng_states, j) * kick_sigma + kick_module
            # application
            t = (1 - t1 ** 2 - t2 ** 2) / (t3 ** 2 + t4 ** 2)
            x[i] += kick * t1
            px[i] += kick * t2
            y[i] += kick * t3 * t
            py[i] += kick * t4 * t

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
            temp1[i] = (px[i] + x[i] * x[i] - y[i] * y[i] + mu *
                        (x[i] * x[i] * x[i] - 3 * x[i] * y[i] * y[i]))
            temp2[i] = (py[i] - 2 * x[i] * y[i] + mu *
                        (3 * x[i] * x[i] * y[i] - y[i] * y[i] * y[i]))

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
def octo_henon_inverse_partial_track(g_x, g_px, g_y, g_py, g_steps, limit, max_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, mu):
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

            px[i] = (px[i] - x[i] * x[i] + y[i] * y[i] - mu *
                        (x[i] * x[i] * x[i] - 3 * x[i] * y[i] * y[i]))
            py[i] = (py[i] + 2 * x[i] * y[i] - mu *
                        (3 * x[i] * x[i] * y[i] - y[i] * y[i] * y[i]))

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
def octo_henon_partial_track_with_kick(g_x, g_px, g_y, g_py, g_steps, limit, max_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, mu, rng_states, kick_module, kick_sigma):
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
            temp1[i] = (px[i] + x[i] * x[i] - y[i] * y[i] + mu *
                        (x[i] * x[i] * x[i] - 3 * x[i] * y[i] * y[i]))
            temp2[i] = (py[i] - 2 * x[i] * y[i] + mu *
                        (3 * x[i] * x[i] * y[i] - y[i] * y[i] * y[i]))

            x[i], px[i] = premade_rotation(
                x[i], temp1[i], omega_x_sin[k], omega_x_cos[k], False)
            y[i], py[i] = premade_rotation(
                y[i], temp2[i], omega_y_sin[k], omega_y_cos[k], False)

            # 4d sphere sampling
            t1 = xoroshiro128p_uniform_float32(rng_states, j) * 2 - 1
            t2 = (xoroshiro128p_uniform_float32(
                rng_states, j) * 2 - 1) * math.sqrt(1 - t1)
            t3 = xoroshiro128p_uniform_float32(rng_states, j) * 2 - 1
            t4 = (xoroshiro128p_uniform_float32(
                rng_states, j) * 2 - 1) * math.sqrt(1 - t3)
                
            # kick module computing
            if kick_sigma == 0:
                kick = kick_module
            else:
                kick = xoroshiro128p_normal_float32(
                    rng_states, j) * kick_sigma + kick_module
            # application
            t = (1 - t1 ** 2 - t2 ** 2) / (t3 ** 2 + t4 ** 2)
            x[i] += kick * t1
            px[i] += kick * t2
            y[i] += kick * t3 * t
            py[i] += kick * t4 * t

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
def octo_henon_inverse_partial_track_with_kick(g_x, g_px, g_y, g_py, g_steps, limit, max_iterations, omega_x_sin, omega_x_cos, omega_y_sin, omega_y_cos, mu, rng_states, kick_module, kick_sigma):
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

            px[i] = (px[i] - x[i] * x[i] + y[i] * y[i] - mu *
                     (x[i] * x[i] * x[i] - 3 * x[i] * y[i] * y[i]))
            py[i] = (py[i] + 2 * x[i] * y[i] - mu *
                     (3 * x[i] * x[i] * y[i] - y[i] * y[i] * y[i]))

            # 4d sphere sampling
            t1 = xoroshiro128p_uniform_float32(rng_states, j) * 2 - 1
            t2 = (xoroshiro128p_uniform_float32(
                rng_states, j) * 2 - 1) * math.sqrt(1 - t1)
            t3 = xoroshiro128p_uniform_float32(rng_states, j) * 2 - 1
            t4 = (xoroshiro128p_uniform_float32(
                rng_states, j) * 2 - 1) * math.sqrt(1 - t3)

            # kick module computing
            if kick_sigma == 0:
                kick = kick_module
            else:
                kick = xoroshiro128p_normal_float32(
                    rng_states, j) * kick_sigma + kick_module
            # application
            t = (1 - t1 ** 2 - t2 ** 2) / (t3 ** 2 + t4 ** 2)
            x[i] += kick * t1
            px[i] += kick * t2
            y[i] += kick * t3 * t
            py[i] += kick * t4 * t

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
