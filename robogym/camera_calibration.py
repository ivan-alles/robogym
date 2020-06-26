#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

"""
Functions for camera calibration tasks, such as camera-robot calibration.
"""

import numpy as np
import scipy
from numpy.linalg import inv
from scipy.spatial.transform import Rotation


def solve_ax_xb_park_martin(a, b):
    """
    Solves AX = XB by the Park-Martin method.

    :param a: list of transformations of the robot in the robot CS (4x4 numpy arrays)
    :param b: list of the transformations of the calibration board in the camera CS (4x4 numpy arrays)
    :return: matrix X, eta1, eta2
       X is the solution
       eta1 is the rotation error, corresponding to the average error (over all measurements and wx, wy, wz
        elementary rotations) in rotation vector (in radians).
       eta2 is the translation error, corresponding to the average error (over all measurements and x, y, z coordinates)
        in coordinates (in given units, e.g. meters).

    See also:
    F. C. Park and B. J. Martin, "Robot sensor calibration: solving AX=XB on the Euclidean group,"
    in IEEE Transactions on Robotics and Automation, vol. 10, no. 5, pp. 717-721, Oct. 1994, doi: 10.1109/70.326576.

    https://www.torsteinmyhre.name/snippets/robcam_calibration.html
    """
    p = len(a)

    if p < 3:
        # for p == 2 an exact solution could be used, but we will not waste time for it.
        raise ValueError("At least three pairs of matrices A and B are required")

    m = np.zeros((3,3), dtype=np.float64)
    log_ra = np.empty((p, 3), dtype=np.float64)
    log_rb = np.empty((p, 3), dtype=np.float64)
    for i in range(p):
        ra = a[i][:3, :3]
        rb = b[i][:3, :3]
        log_ra[i] = Rotation.from_matrix(ra).as_rotvec()
        log_rb[i] = Rotation.from_matrix(rb).as_rotvec()

    for i in range(p):
        m += np.outer(log_rb[i], log_ra[i])

    rx = np.dot(scipy.linalg.sqrtm(inv((np.dot(m.T, m)))), m.T)

    err1 = 0
    for i in range(p):
        e = rx.dot(log_rb[i]) - log_ra[i]  # e is a vector of 3 elements
        err1 += (e * e).sum()
    eta1 = np.sqrt(err1 / (3*p))

    c = np.zeros((3*p, 3), dtype=np.float64)
    d = np.zeros((3*p, 1), dtype=np.float64)
    for i in range(p):
        ra, ta = a[i][:3, :3], a[i][:3, 3]
        rb, tb = b[i][:3, :3], b[i][:3, 3]
        c[3*i:3*i+3] = np.eye(3, dtype=np.float64) - ra
        d[3*i:3*i+3, 0] = ta - np.dot(rx, tb)

    tx = np.linalg.lstsq(c, d, rcond=None)[0]
    err2 = c.dot(tx) - d
    eta2 = np.sqrt((err2 * err2).sum() / len(d))

    x = np.eye(4, dtype=np.float64)
    x[:3, :3] = rx
    x[:3, 3:] = tx

    return x, eta1, eta2


def calibrate_camera_to_robot(robot_poses_tool, camera_poses_marker):
    """
    Calibrates camera to robot. Calibration object is mounted on the robot tool.

    :param robot_poses_tool: list of tool poses in the robot CS (4x4 numpy arrays).
    :param camera_poses_marker: list of marker poses in the camera CS (4x4 numpy arrays).
    :return: robot_pose_camera, tool_pose_marker, e1, e2.
    robot_pose_camera is the camera pose in the robot CS.
    tool_pose_marker is the marker pose in the tool CS.
    e1 is the calibration error for rotation (in radians).
    e2 is the calibration error for translation (in input units).
    """
    a = []  # List of deltas of gripper pose in robot CS
    b = []  # List of deltas of calibration object pose in camera CS

    poses_count = len(camera_poses_marker)

    if len(robot_poses_tool) != poses_count:
        raise ValueError("Number of gripper poses must match number of calibration object poses")

    for i in range(poses_count):
        for j in range(i + 1, poses_count):
            a.append(get_pose_delta(robot_poses_tool[i], robot_poses_tool[j]))

            b.append(get_pose_delta(camera_poses_marker[i], camera_poses_marker[j]))

    x, e1, e2 = solve_ax_xb_park_martin(a, b)

    tool_pose_marker = x

    # TODO(ia): compute average from all measurements.
    robot_pose_camera = np.dot(np.dot(robot_poses_tool[0], tool_pose_marker), inv(camera_poses_marker[0]))

    return robot_pose_camera, tool_pose_marker, e1, e2


def get_pose_delta(p0, p1):
    d = np.dot(inv(p0), p1)
    return d
