import pytest

import numpy as np
import henon_map as hm

dr = 0.1
epsilon = 16.0
alpha = np.array([1.233731073461239])
th1 = np.array([0.0])
th2 = np.array([0.0])

scan_list = np.array([10000,10])

np.set_printoptions(precision=15)

def test_equality():
    engine1 = hm.radial_scan.generate_instance(dr, alpha, th1, th2, epsilon)

    radius = engine1.compute(scan_list)
    print(radius)

    engine2 = hm.full_track.generate_instance(
        np.array([0.1 * 4]), alpha, th1, th2, np.array([10000]), epsilon)

    x,px,y,py = engine2.compute()

    engine3 = hm.partial_track.generate_instance(
        np.array([0.1 * 4]), alpha, th1, th2, epsilon)

    # print("comparison", x[0], engine3.x)
    # print("comparison", y[0], engine3.y)
    # print("comparison", px[0], engine3.px)
    # print("comparison", py[0], engine3.py)
    # for i in range(10000):
    #     c_x, c_px, c_y, c_py, step = engine3.compute(1)
    #     print("comparison {}".format(i+1), x[i+1], c_x)
    #     print("comparison {}".format(i+1), px[i+1], c_px)
    #     print("comparison {}".format(i+1), y[i+1], c_y)
    #     print("comparison {}".format(i+1), py[i+1], c_py)

    #     assert c_x == x[i + 1], str(i + 1)

    print(radius[0,0])
    assert np.count_nonzero(np.isnan(x)) == 0, "coso"
    assert np.count_nonzero(np.isnan(px)) == 0, "coso"
    assert np.count_nonzero(np.isnan(y)) == 0, "coso"
    assert np.count_nonzero(np.isnan(py)) == 0, "coso"
    assert False
