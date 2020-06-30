#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group

from robogym import camera_calibration


def test_solve_ax_xb_park_martin():
    rng = np.random.RandomState(seed=1)

    x = random_rigid_transform_3d(rng)

    a = []
    b = []

    for i in range(10):
        a_i = random_rigid_transform_3d(rng)
        b_i = inv(x).dot(a_i).dot(x)
        a.append(a_i)
        b.append(b_i)

    x1, e1, e2 = camera_calibration.solve_ax_xb_park_martin(a, b)

    assert e1 < 1e-5
    assert e2 < 1e-5
    assert np.linalg.norm(x1 - x) < 1e-4

    # now generate noisy data
    x = random_rigid_transform_3d(rng)
    x1 = solve_ax_xb_noisy(x, 3, 0.01, rng)
    assert np.linalg.norm(x1 - x) < 0.03

    x1 = solve_ax_xb_noisy(x, 10, 0.01, rng)
    assert np.linalg.norm(x1 - x) < 0.033

    x1 = solve_ax_xb_noisy(x, 100, 0.01, rng)
    assert np.linalg.norm(x1 - x) < 0.0056


def test_calibrate_camera_to_robot():
    rng = np.random.RandomState(seed=1)
    robot_pose_camera = random_rigid_transform_3d(rng)  # Camera pose in robot CS
    tool_pose_marker = random_rigid_transform_3d(rng)  # Pose of the marker on the tool

    robot_poses_tool = []  # Poses of the tool in robot CS
    camera_poses_marker = []  # Poses of the calibration object in camera CS

    for i in range(10):
        robot_pose_tool = random_rigid_transform_3d(rng)  # Random tool pose in robot CS
        robot_pose_marker = np.dot(robot_pose_tool, tool_pose_marker)  # Pose of the marker object in robot CS.
        camera_pose_marker = np.dot(inv(robot_pose_camera), robot_pose_marker)  # Pose of the marker in camera CS.

        robot_poses_tool.append(robot_pose_tool)
        camera_poses_marker.append(camera_pose_marker)

    robot_pose_camera_1, tool_pose_marker_1, e1, e2 = camera_calibration.calibrate_camera_to_robot(robot_poses_tool,
                                                                                                   camera_poses_marker)

    assert e1 < 1e-5
    assert e2 < 1e-5
    assert np.linalg.norm(robot_pose_camera_1 - robot_pose_camera) < 1e-4
    assert np.linalg.norm(tool_pose_marker_1 - tool_pose_marker) < 1e-4


def solve_ax_xb_noisy(x, sample_count, error_magnitude, rng):
    a = []
    b = []

    log_xr = Rotation.from_matrix(x[0:3, 0:3]).as_rotvec()

    x_n = np.zeros((4, 4))
    x_n[3, 3] = 1

    for i in range(sample_count):
        a_i = random_rigid_transform_3d(rng)
        log_xr_n = log_xr + rng.randn(3) * (error_magnitude * log_xr)
        xr_n = Rotation.from_rotvec(log_xr_n).as_matrix()
        xt_n = x[0:3, 3:4] + rng.randn(3, 1) * (error_magnitude * x[0:3, 3:4])

        x_n[0:3, 0:3] = xr_n
        x_n[0:3, 3:4] = xt_n

        b_i = inv(x_n).dot(a_i).dot(x_n)
        a.append(a_i)
        b.append(b_i)

    x1, e1, e2 = camera_calibration.solve_ax_xb_park_martin(a, b)
    return x1


def random_rigid_transform_3d(rng):
    t = np.zeros((4, 4))
    t[0:3, 0:3] = special_ortho_group.rvs(dim=3, random_state=rng)
    t[0:3, 3:] = rng.rand(3, 1)
    t[3, 3] = 1
    return t
