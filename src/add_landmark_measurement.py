import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    # Get X(4) from initial_estimate (not result, since we haven't re-optimized yet)
    pose4 = initial_estimate.atPose2(X(4))
    # Get L(2) from result (it was optimized in the previous step)
    landmark2 = result.atPoint2(L(2))

    # Compute range (straight-line distance)
    dx = landmark2[0] - pose4.x()
    dy = landmark2[1] - pose4.y()
    distance = math.sqrt(dx**2 + dy**2)

    # Compute bearing relative to robot's heading
    global_angle = math.atan2(dy, dx)
    bearing = global_angle - pose4.theta()

    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2(bearing), distance, MEASUREMENT_NOISE))
    return graph