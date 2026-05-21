import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    return result





def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_trace = float('inf')
    best_sum = float('inf')

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = graph.clone()
            ie = gtsam.Values(initial_estimate)
            g, ie = add_pose(g, ie, pose_5)
            result = optimize(g, ie)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, result)

            marginals = gtsam.Marginals(g, result)
            trace = (np.trace(marginals.marginalCovariance(L(1))) +
                     np.trace(marginals.marginalCovariance(L(2))))
            s = (marginals.marginalCovariance(L(1)).sum() +
                 marginals.marginalCovariance(L(2)).sum())

            if trace < best_trace:
                best_trace = trace
                best_sum = s
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_sum



def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_error = float('inf')
    best_mahal = float('inf')

    true_poses = [gtsam.Pose2(0,0,0), gtsam.Pose2(2,0,0), gtsam.Pose2(4,0,0)]

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            g = graph.clone()
            ie = gtsam.Values(initial_estimate)
            g, ie = add_pose(g, ie, pose_5)
            result = optimize(g, ie)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, result)

            error = g.error(result)

            marginals = gtsam.Marginals(g, result)
            mahal = 0
            for i in range(1, 4):
                cov = marginals.marginalCovariance(X(i))
                est = result.atPose2(X(i))
                true = true_poses[i-1]
                diff = np.array([est.x()-true.x(), est.y()-true.y(), est.theta()-true.theta()])
                mahal += diff @ np.linalg.inv(cov) @ diff

            if error < best_error:
                best_error = error
                best_mahal = mahal
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_mahal