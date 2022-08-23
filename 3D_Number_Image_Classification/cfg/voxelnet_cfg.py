import math

class config:
    # batch size
    N=2

    # device
    device="cuda"

    # maxiumum number of points per voxel
    T=35

    # voxel size
    vd = 0.4
    vh = 0.2
    vw = 0.2

    # Traing points cloud range
    # xrange = (-0.1579, 0.1579)
    # yrange = (-0.7023, 0.6843)
    # zrange = (-0.6795, 0.6597)

    # Test Point cloud range
    xrange = (-0.7839, 0.7732)
    yrange = (-0.7591, 0.7707)
    zrange = (-0.7849, 0.7697)

    # voxel grid
    W = math.ceil((xrange[1] - xrange[0]) / vw)
    H = math.ceil((yrange[1] - yrange[0]) / vh)
    D = math.ceil((zrange[1] - zrange[0]) / vd)
