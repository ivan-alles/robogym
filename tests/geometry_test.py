#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import math
import numpy as np

from robogym import geometry


def test_normalize_angle():
    assert geometry.normalize_angle(0, 360, 0) == 0
    assert geometry.normalize_angle(200., 360, 0) == 200
    assert geometry.normalize_angle(200., 360) == -160.0
    assert geometry.normalize_angle(200, 360, -180) == -160.0

    assert geometry.normalize_angle(100, 180) == -80

    assert np.allclose(geometry.normalize_angle(np.pi + 0.1, 2 * np.pi), -np.pi + 0.1)
    assert np.allclose(geometry.normalize_angle(np.pi + 0.1), -np.pi + 0.1)
    assert np.allclose(geometry.normalize_angle(np.pi + 0.1, 2 * np.pi, -np.pi), -np.pi + 0.1)


def test_mean_of_angles():
    m_act = geometry.mean_of_angles(np.array([0.0, 0, 0]))
    assert np.allclose(0, m_act)

    m_act = geometry.mean_of_angles(np.array([0.1, 0.1, 0.1, 0.1]))
    assert np.allclose(0.1, m_act)

    m_act = geometry.mean_of_angles(np.array([0.1, -0.1, 0.1, -0.1]))
    assert np.allclose(0.0, m_act)

    m_act = geometry.mean_of_angles(np.array([0, 2*np.pi]))
    assert np.allclose(0.0, m_act)

    m_act = geometry.mean_of_angles(np.array([0, 2 * np.pi, -10 * np.pi]))
    assert np.allclose(0.0, m_act)

    m_act = geometry.mean_of_angles(np.array([
        [0.1, 0.1 + 2 * np.pi, 0.1 - 10 * np.pi],
        [-0.1, -0.1 + 2 * np.pi, -0.1 - 10 * np.pi]
    ]), axis=1)
    assert np.allclose([0.1, -0.1], m_act)

    m_act = geometry.mean_of_angles(np.array([
        [0.1, -0.1 + 2 * np.pi, -0.1 - 10 * np.pi, 0.1 + 8 * np.pi],
        [np.pi, 3*np.pi, -5*np.pi, 9*np.pi],
    ]), axis=1)
    assert np.allclose([0, np.pi], m_act)


def test_line_by_points():
    # Make sure the line is correctly oriented.
    line = geometry.line_by_points((0, 0), (1, 0))
    assert np.allclose((0, 1, 0), line)

    s2 = math.sqrt(2) / 2

    line = geometry.line_by_points((0, 1), (1, 0))
    assert np.allclose((s2, s2, -s2), line)

    line = geometry.line_by_points((0, 1), (0, 1))
    assert line is None


def test_line_point_distance():
    line = [1, 0, 2]
    points = [[0, 0], [1, 0], [0, 1], [-2, 1], [-3, -1]]
    assert np.allclose([[2, 3, 2, 0, -1]], geometry.line_point_distance(line, points))


def test_line_intersection():
    c = geometry.line_intersection((0, 1, 0), (-1, -1, 1))
    assert np.allclose([1, 0], c)

    c = geometry.line_intersection((0, 1, 0), (0, 1, 1))
    assert c is None

    c = geometry.line_intersection((0, 1, 0), (0, 1, 0))
    assert c is None


def test_plane_by_normal_point():
    p = geometry.plane_by_normal_point((0, 0, 1), (0, 0, 0))
    assert np.allclose(p, (0, 0, 1, 0))

    p = geometry.plane_by_normal_point((0, 0, 1), (0, 0, 5))
    assert np.allclose(p, (0, 0, 1, -5))

    p = geometry.plane_by_normal_point((0, 0, -1), (0, 0, 5))
    assert np.allclose(p, (0, 0, -1, 5))

    p = geometry.plane_by_normal_point((0, 0, 2), (0, 0, 5))
    assert np.allclose(p, (0, 0, 1, -5))


def test_plane_line_intersection():
    intersection_point = geometry.plane_line_intersection((0, 0, 1), (0, 0, 0), (0, 0, -1), (0, 0, 1))
    assert np.allclose(intersection_point, (0, 0, 0))

    intersection_point = geometry.plane_line_intersection((0, 0, -1), (0, 0, 1), (0, 0, -1), (2, 0, 3))
    assert np.allclose(intersection_point, (1, 0, 1))

    intersection_point = geometry.plane_line_intersection((0, 0, 1), (0, 0, 1), (0, 0, -1), (2, 0, 3))
    assert np.allclose(intersection_point, (1, 0, 1))

    intersection_point = geometry.plane_line_intersection((0, 0, 1), (0, 11, 1), (0, 22, -1), (2, 22, 3))
    assert np.allclose(intersection_point, (1, 22, 1))

    intersection_point = geometry.plane_line_intersection((0, 0, 1), (0, 0, 1), (0, 0, 1), (5, 0, 3))
    assert np.allclose(intersection_point, (0, 0, 1))

    intersection_point = geometry.plane_line_intersection((0, 0, 1), (0, 0, 2), (0, 0, 1), (0, 0, 1))
    assert intersection_point is None


def test_spherical_to_cartesian_n_2():
    def compute(r, phi):
        exp = [
            r * np.cos(phi[0]),
            r * np.sin(phi[0])
        ]
        return exp

    r = 1.
    phi = np.array([0.1])
    act = geometry.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 0.
    phi = np.array([0.1])
    act = geometry.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 0.
    phi = np.array([0.2])
    act = geometry.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 3.
    phi = np.array([0.1])
    act = geometry.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 2.
    phi = np.array([0.3])
    act = geometry.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 2.
    phi = np.array([-0.3])
    act = geometry.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)


def test_spherical_to_cartesian_n_3():
    def compute(r, phi):
        exp = [
            r * np.cos(phi[0]),
            r * np.sin(phi[0]) * np.cos(phi[1]),
            r * np.sin(phi[0]) * np.sin(phi[1])
        ]
        return np.array(exp).ravel()
    r = 1.
    phi = np.array([0.1, 0.2])
    act = geometry.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 2.
    phi = np.array([0.1, 0.2])
    act = geometry.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = np.array([3., 4., 5]).reshape(-1, 1)
    phi = np.array([[0.1, -0.2], [0.1, 0.2], [0.3, -0.2]])
    act = geometry.spherical_to_cartesian_n(r, phi)
    exp0 = compute(r[0], phi[0])
    exp1 = compute(r[1], phi[1])
    exp2 = compute(r[2], phi[2])
    assert np.allclose(exp0, act[0])
    assert np.allclose(exp1, act[1])
    assert np.allclose(exp2, act[2])


def convert_cartesian_to_spherical(x):
    r = np.linalg.norm(x)
    n = len(x)
    phi = np.zeros(n - 1)
    k = -1
    for i in range(n - 1, -1, -1):
        if x[i] != 0:
            k = i
            break
    if k == -1:
        return r, phi

    for i in range(0, k):
        phi[i] = np.arccos(x[i] / np.linalg.norm(x[i:]))

    if k == n - 1:
        phi[n - 2] = np.arccos(x[n-2] / np.linalg.norm(x[-2:]))
        if x[n-1] < 0:
            phi[n - 2] *= -1
    else:
        phi[k] = 0 if x[k] > 0 else np.pi

    return r, phi


def test_cartesian_to_spherical_n_2():
    x = np.array([
        [0, 0],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -2],
        [2, 3],
        [-1, 4]
    ], dtype=np.float32)

    r_act, phi_act = geometry.cartesian_to_spherical_n(x)

    for i in range(len(x)):
        r_exp, phi_exp = convert_cartesian_to_spherical(x[i])
        assert np.allclose(r_act[i], r_exp)
        assert np.allclose(phi_act[i], phi_exp)


def test_cartesian_to_spherical_n_3():
    x = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [2, 1, 0],
        [2, -1, 0],
        [-2, -1, 0],
        [2, 3, 1],
        [2, 3, -1],
        [2, 0, 5],
        [0, 0, 1],
        [0, 0, -2]
    ], dtype=np.float32)

    r_act, phi_act = geometry.cartesian_to_spherical_n(x)

    for i in range(len(x)):
        r_exp, phi_exp = convert_cartesian_to_spherical(x[i])
        assert np.allclose(r_act[i], r_exp)
        assert np.allclose(phi_act[i], phi_exp)


def test_cartesian_to_spherical_n_4():
    x = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [1, 2, 0, 0],
        [-1, -2, 0, 0],
        [2, 2, 1, 0],
        [-1, -2, -3, 0],
        [2, 2, 1, -1],
        [-1, -2, -3, 4],
    ], dtype=np.float32)

    r_act, phi_act = geometry.cartesian_to_spherical_n(x)

    for i in range(len(x)):
        r_exp, phi_exp = convert_cartesian_to_spherical(x[i])
        assert np.allclose(r_act[i], r_exp)
        assert np.allclose(phi_act[i], phi_exp)


def test_cartesian_to_spherical_n_back_and_forth():
    rng = np.random.RandomState(seed=1)

    for r in range(10000):
        n = rng.randint(2, 100)
        if rng.uniform(0, 1) < .1:
            x = rng.uniform(-10, 10, n)  # row-vector
        else:
            m = rng.randint(1, 10)
            x = rng.uniform(-10, 10, (m, n))  # array of row vectors
        r, phi = geometry.cartesian_to_spherical_n(x)
        if phi.ndim == 1:
            phi = phi.reshape(1, -1)

        assert phi.shape[1] == n - 1
        # Last phi is in range -pi, pi
        assert np.logical_and(phi[:, -1] >= -np.pi, phi[:, -1] <= np.pi).sum() == len(phi)
        # Other phi are in range [0, pi]
        if n > 2:
            assert np.logical_and(phi[:, :-1] >= 0, phi[:, :-1] <= np.pi).sum() == len(phi) * (n-2)

        x1 = geometry.spherical_to_cartesian_n(r, phi)
        assert np.allclose(x, x1)

        # Make sure that spherical -> cartesian transform is invariant to
        # the following angle transformations:

        phi[:, -1] += rng.randint(-10, 10) * 2 * np.pi

        x2 = geometry.spherical_to_cartesian_n(r, phi)
        assert np.allclose(x, x2)
