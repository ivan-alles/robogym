#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import math

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation

import robogym.geometry as geo
import robogym.transform3 as t3
from robogym.transform3 import Transform3


class Transform3Test:
    def test___init__(self):
        assert np.allclose(Transform3().m, np.identity(4))

        m = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], np.float32).reshape(4, 4)

        # 4x4 matrix
        assert np.allclose(Transform3(m).m, m)
        assert np.allclose(Transform3(np.ravel(m)).m, m)

        # 3x3 matrix
        tr = Transform3(m[0:3, 0:3])
        assert np.allclose(tr.m[0:3, 0:3], m[0:3, 0:3])
        assert np.allclose(tr.m[0:3, 3:], np.zeros((3, 1)))
        assert np.allclose(tr.m[3:, 0:], [0, 0, 0, 1])
        tr = Transform3(np.ravel(m[0:3, 0:3]))
        assert np.allclose(tr.m[0:3, 0:3], m[0:3, 0:3])
        assert np.allclose(tr.m[0:3, 3:], np.zeros((3, 1)))
        assert np.allclose(tr.m[3:, 0:], [0, 0, 0, 1])

        # 3x4 matrix
        tr = Transform3(m[0:3, 0:4])
        assert np.allclose(tr.m[0:3, 0:4], m[0:3, 0:4])
        assert np.allclose(tr.m[3:, 0:], [0, 0, 0, 1])
        tr = Transform3(np.ravel(m[0:3, 0:4]))
        assert np.allclose(tr.m[0:3, 0:4], m[0:3, 0:4])
        assert np.allclose(tr.m[3:, 0:], [0, 0, 0, 1])

        # 3 vector, 3x3 matrix,
        tr = Transform3(m[0:3, 3:].squeeze(), m[0:3, 0:3])
        assert np.allclose(tr.m[0:3, 0:4], m[0:3, 0:4])
        assert np.allclose(tr.m[3:, 0:], [0, 0, 0, 1])

        # 3 tvec
        m = np.identity(4)
        tvec = np.array([11.0, 12.0, 13.0]).reshape(3, 1)
        m[0:3, 3:] = tvec
        assert np.allclose(Transform3(tvec).m, m)
        assert np.allclose(Transform3.from_t(tvec).m, m)

        # 3 tvec, 3 rvec
        a = 0.1
        ca = math.cos(a)
        sa = math.sin(a)
        r = np.array([ca, -sa, 0, sa, ca, 0,  0, 0, 1]).reshape(3, 3)
        tvec = np.array([1, 2, 3])
        rvec = np.array([0, 0, 1]) * a
        tr = Transform3(tvec, rvec)
        assert np.allclose(tr.m[0:3, 0:3], r)
        assert np.allclose(tr.m[0:3, 3:], tvec.reshape(3, 1))
        assert np.allclose(tr.m[3:, 0:], [0, 0, 0, 1])

        # 6 DOF vector (tvec, rvec)
        tr = Transform3(np.concatenate((tvec, rvec)))
        assert np.allclose(tr.m[0:3, 0:3], r)
        assert np.allclose(tr.m[0:3, 3:], tvec.reshape(3, 1))
        assert np.allclose(tr.m[3:, 0:], [0, 0, 0, 1])

        # Center, axis, angle
        a = 0.2
        axis = np.array([1.0, 2, 3])
        center = np.array([10.0, 15, 20]).reshape(3, 1)
        t1 = np.identity(4)
        t1[0:3, 3:] = -center
        r = np.identity(4)
        r[0:3, 0:3] = Rotation.from_rotvec(axis / norm(axis) * a).as_matrix()
        t2 = np.identity(4)
        t2[0:3, 3:] = center
        m = np.linalg.multi_dot((t2, r, t1))
        tr = Transform3.from_center_axis_angle(center, axis, a)
        assert np.allclose(tr.m, m)

        # Scalar
        tr = Transform3(2.0)
        assert np.allclose(tr.m, np.array([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]).reshape(4, 4))
        tr = Transform3.from_scale(2.0)
        assert np.allclose(tr.m, np.array([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1]).reshape(4, 4))
        tr = Transform3.from_scale([2, 0.5, 3])
        assert np.allclose(tr.m, np.array([2, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1]).reshape(4, 4))

        # String
        tr = Transform3('0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15')
        assert np.allclose(tr.m, np.arange(16).reshape(4, 4))
        tr = Transform3.from_str('0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15')
        assert np.allclose(tr.m, np.arange(16).reshape(4, 4))

    def test_item(self):
        tr = Transform3()
        assert tr[1, 1] == 1
        assert tr[0, 1] == 0
        assert np.allclose(tr[0:2, 0:2], np.identity(2))
        tr[0, 1] = 123
        assert tr.m[0, 1] == 123
        m = np.ones((2, 2))
        tr[1:3, 1:3] = m
        assert np.allclose(tr[1:3, 1:3], m)

    def test_accessors(self):
        tr = Transform3()
        t = [10, 20, 30]
        ax = [-1, 0, 0]
        ay = [0, 1, 0]
        az = [0, 0, -1]
        tr.t = t
        tr.ax = ax
        tr.ay = ay
        tr.az = az
        assert np.allclose(t, tr.t)
        assert np.allclose(ax, tr.ax)
        assert np.allclose(ay, tr.ay)
        assert np.allclose(az, tr.az)
        tr.r = np.eye(3)
        assert np.allclose(tr.r, np.eye(3))

    def test_pose6(self):
        m = np.array([0, -1, 0, 11, 1, 0, 0, 12,  0, 0, 1.0, 13, 0, 0, 0, 1]).reshape(4, 4)
        tr = Transform3(m)
        pose6 = tr.pose6()
        assert np.allclose(pose6, np.array([11.0, 12, 13, 0, 0, np.pi / 2]))

    def test_t_axis_angle(self):
        t = [1, 2, 3]
        axis = [1, 0, 0]
        angle = 0.5
        tr = Transform3.from_t_axis_angle(t, axis, angle)
        t1, axis1, angle1 = tr.t_axis_angle()
        assert np.allclose(t, t1)
        assert np.allclose(axis, axis1)
        assert np.allclose(angle, angle1)

    def test_t3_chain(self):
        rng = np.random.RandomState(seed=1)

        m1 = rng.random_sample((4, 4))
        m2 = rng.random_sample((4, 4))
        m3 = rng.random_sample((4, 4))

        result = t3.chain(m1, m2)
        assert type(result) == Transform3
        assert np.allclose(result.m, np.linalg.multi_dot([m1, m2]))

        result = t3.chain(Transform3(m1), m2)
        assert type(result) == Transform3
        assert np.allclose(result.m, np.linalg.multi_dot([m1, m2]))

        result = t3.chain(m1, Transform3(m2), m3)
        assert type(result) == Transform3
        assert np.allclose(result.m, np.linalg.multi_dot([m1, m2, m3]))

    def test_t3_dot(self):
        rng = np.random.RandomState(seed=1)

        m1 = rng.random_sample((4, 4))
        m2 = rng.random_sample((4, 4))
        m3 = rng.random_sample((4, 4))
        v = rng.random_sample((4, 32))

        # Result is a new Transform()
        result = t3.dot(m1, Transform3(m2), m3)
        assert type(result) == np.ndarray
        assert np.allclose(result, np.linalg.multi_dot([m1, m2, m3]))

        # Result is a transformed array
        result = t3.dot(m1, Transform3(m2), m3, v)
        assert type(result) == np.ndarray
        assert np.allclose(result, np.linalg.multi_dot([m1, m2, m3, v]))

    def test_dot(self):
        # Some scaling and translation
        tr = Transform3(np.array([[1, 0, 0, 10], [0, 2, 0, 20], [0, 0, 3, 30], [0, 0, 0, 1]], np.float32))

        # Only scale
        tr1 = Transform3(np.array([[5, 0, 0, 0], [0, 10, 0, 0],  [0, 0, 20, 0],  [0, 0, 0, 1]], np.float32))

        exp_r = np.array(np.array([[5, 0, 0, 10], [0, 20, 0, 20],  [0, 0, 60, 30],  [0, 0, 0, 1]], np.float32))
        r = tr.dot(tr1)
        assert type(r) == Transform3
        assert self._check_dim_and_equality(r.m, exp_r)
        # Matrix 4x4
        assert self._check_dim_and_equality(tr.dot(tr1.m), exp_r)

        # Array of vectors

        v3 = np.array([1, 2, 3], np.float32)
        exp_r = np.array([11, 24, 39], np.float32)
        assert self._check_dim_and_equality(tr.dot(v3), exp_r)
        assert self._check_dim_and_equality(tr.dot(v3.T, transpose=True), exp_r.T)
        assert self._check_dim_and_equality(tr.dot(v3.reshape(3, 1)), exp_r.reshape(3, 1))
        assert self._check_dim_and_equality(tr.dot(v3.reshape(3, 1).T, True), exp_r.reshape(3, 1).T)

        v4 = np.append(v3, 1)
        exp_r = np.append(exp_r, 1)
        assert self._check_dim_and_equality(tr.dot(v4), exp_r)
        assert self._check_dim_and_equality(tr.dot(v4.T, True), exp_r.T)
        assert self._check_dim_and_equality(tr.dot(v4.reshape(4, 1)), exp_r.reshape(4, 1))
        assert self._check_dim_and_equality(tr.dot(v4.reshape(4, 1).T, True), exp_r.reshape(4, 1).T)

        v35 = np.array([[1, 2, 3], [1, 1, 1], [0, 0, 0], [3, 2, 1], [2, 2, 2]], np.float32).T
        exp_r = np.array([[11, 24, 39], [11, 22, 33], [10, 20, 30], [13, 24, 33], [12, 24, 36]], np.float32).T
        assert self._check_dim_and_equality(tr.dot(v35), exp_r)
        assert self._check_dim_and_equality(tr.dot(v35.T, True), exp_r.T)

        v45 = np.append(v35, np.full((1, 5), 1, np.float32), axis=0)
        exp_r = np.append(exp_r, np.full((1, 5), 1, np.float32), axis=0)
        assert self._check_dim_and_equality(tr.dot(v45), exp_r)
        assert self._check_dim_and_equality(tr.dot(v45.T, True), exp_r.T)

    def test_project_unproject(self):
        # A camera intrinsics matrix
        k = Transform3([[800, 0, 320, 0], [0, 800, 240, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        p3 = np.array([[0, 0, 1], [0.1, 0.3, 1], [0.2, 0.6, 2], [0.1, 0.3, 4]], np.float32)
        p2 = k.project(p3, True)
        exp_p2 = np.array([[320, 240], [400, 480], [400, 480], [340, 300]], np.float32)
        assert self._check_dim_and_equality(p2, exp_p2)

        k_1 = k.inv()
        p3_1 = k_1.unproject(p2, True)

        # Scale by z to obtain original point
        p3_1 = np.multiply(p3_1, p3[:, 2:])
        assert self._check_dim_and_equality(p3_1, p3, atol=1.e-5)

    def test_unproject_on_plane(self):
        # A camera intrinsics matrix
        k = Transform3([[800, 0, 320, 0], [0, 800, 240, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # Our will be the xy plane with z-axis as a normal vector.
        # Define it by a rotation around x-axis and a translation along z-axis.
        camera_pose_plane = Transform3.from_t_axis_angle([0, 0, 1], [1, 0, 0], 2.7)
        # All points are in xy plane
        plane_points = np.array([[0, 0, 0], [0.1, 0.2, 0], [-0.3, -0.2, 0], [-0.1, 0.2, 0]])

        camera_points = camera_pose_plane.dot(plane_points, True)
        image_points = k.project(camera_points, True)
        plane_eq = geo.plane_by_normal_point(camera_pose_plane.az, camera_pose_plane.t)
        k_1 = k.inv()
        points_camera_1 = k_1.unproject_on_plane(image_points, plane_eq, True)
        assert self._check_dim_and_equality(camera_points, points_camera_1, atol=1.e-5)

    def test_str_pose6(self):
        tr = Transform3.from_pose6([1, 2, 3, 0, 2, 0])
        text = tr.str_pose6(format='{:.1f} {:.1f} {:.1f}   {:.2f} {:.2f} {:.2f}')
        assert text == "1.0 2.0 3.0   0.00 2.00 0.00"

    def test_str_t_axis_angle(self):
        tr = Transform3.from_t_axis_angle([1, 2, 3], [0, 1, 0], 2)
        text = tr.str_t_axis_angle(format='{:.1f} ' * 7)
        assert text == "1.0 2.0 3.0 0.0 1.0 0.0 2.0 "

    def _check_dim_and_equality(self, a1, a2, rtol=1.e-5, atol=1.e-8, equal_nan=False):
        if a1.ndim != a2.ndim:
            return False
        for d in range(a1.ndim):
            if a1.shape[d] != a2.shape[d]:
                return False
        return np.allclose(a1, a2, rtol=rtol, atol=atol, equal_nan=equal_nan)
