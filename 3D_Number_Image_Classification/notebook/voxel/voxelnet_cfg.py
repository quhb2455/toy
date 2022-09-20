import math
import numpy as np

class config:
    # batch size
    N=64

    # device
    device="cuda"

    # maxiumum number of points per voxel
    T=42

    # voxel size
    vd = 0.05
    vh = 0.05
    vw = 0.05

    # Training set points cloud range
    # train_xrange = (-0.1579, 0.1579)
    # train_yrange = (-0.7023, 0.6843)
    # train_zrange = (-0.6795, 0.6597)
    train_range = np.array([(-0.1579, 0.1579), (-0.7023, 0.6843), (-0.6795, 0.6597)])

    # Test set Point cloud range
    # test_xrange = (-0.7839, 0.7732)
    # test_yrange = (-0.7591, 0.7707)
    # test_zrange = (-0.7849, 0.7697)
    test_range = np.array([(-0.7839, 0.7732), (-0.7591, 0.7707), (-0.7849, 0.7697)])

    # voxel grid
    W = math.ceil((test_range[0][1] - test_range[0][0]) / vw)
    H = math.ceil((test_range[1][1] - test_range[1][0]) / vh)
    D = math.ceil((test_range[2][1] - test_range[2][0]) / vd)
