#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import math

from robogym.geometry import *


def test_normalize_angle():
    assert normalize_angle(0, 360, 0) == 0
    assert normalize_angle(200., 360, 0) == 200
    assert normalize_angle(200., 360) == -160.0
    assert normalize_angle(200, 360, -180) == -160.0

    assert normalize_angle(100, 180) == -80

    assert np.allclose(normalize_angle(np.pi + 0.1, 2 * np.pi), -np.pi + 0.1)
    assert np.allclose(normalize_angle(np.pi + 0.1), -np.pi + 0.1)
    assert np.allclose(normalize_angle(np.pi + 0.1, 2 * np.pi, -np.pi), -np.pi + 0.1)


def test_mean_of_angles():
    m_act = mean_of_angles(np.array([0.0, 0, 0]))
    assert np.allclose(0, m_act)

    m_act = mean_of_angles(np.array([0.1, 0.1, 0.1, 0.1]))
    assert np.allclose(0.1, m_act)

    m_act = mean_of_angles(np.array([0.1, -0.1, 0.1, -0.1]))
    assert np.allclose(0.0, m_act)

    m_act = mean_of_angles(np.array([0, 2*np.pi]))
    assert np.allclose(0.0, m_act)

    m_act = mean_of_angles(np.array([0, 2 * np.pi, -10 * np.pi]))
    assert np.allclose(0.0, m_act)

    m_act = mean_of_angles(np.array([
        [0.1, 0.1 + 2 * np.pi, 0.1 -10 * np.pi],
        [-0.1, -0.1 + 2 * np.pi, -0.1 - 10 * np.pi]
    ]), axis=1)
    assert np.allclose([0.1, -0.1], m_act)

    m_act = mean_of_angles(np.array([
        [0.1, -0.1 + 2 * np.pi, -0.1 - 10 * np.pi, 0.1 + 8 * np.pi],
        [np.pi, 3*np.pi, -5*np.pi, 9*np.pi],
    ]), axis=1)
    assert np.allclose([0, np.pi], m_act)


def test_line_by_points():
    # Make sure the line is correctly oriented.
    line = line_by_points((0, 0), (1, 0))
    assert np.allclose((0, 1, 0), line)

    s2 = math.sqrt(2) / 2

    line = line_by_points((0, 1), (1, 0))
    assert np.allclose((s2, s2, -s2), line)

    line = line_by_points((0, 1), (0, 1))
    assert line is None


def test_line_point_distance():
    line = [1, 0, 2]
    points = [[0, 0], [1, 0], [0, 1], [-2, 1], [-3, -1]]
    assert np.allclose([[2, 3, 2, 0, -1]], line_point_distance(line, points))


def test_line_intersection():
    c = line_intersection((0, 1, 0), (-1, -1, 1))
    assert np.allclose([1, 0], c)

    c = line_intersection((0, 1, 0), (0, 1, 1))
    assert c is None

    c = line_intersection((0, 1, 0), (0, 1, 0))
    assert c is None

def test_plane_by_normal_point():
    p = plane_by_normal_point((0, 0, 1), (0, 0, 0))
    assert np.allclose(p, (0, 0, 1, 0))

    p = plane_by_normal_point((0, 0, 1), (0, 0, 5))
    assert np.allclose(p, (0, 0, 1, -5))

    p = plane_by_normal_point((0, 0, -1), (0, 0, 5))
    assert np.allclose(p, (0, 0, -1, 5))

    p = plane_by_normal_point((0, 0, 2), (0, 0, 5))
    assert np.allclose(p, (0, 0, 1, -5))


def test_plane_line_intersection():
    intersection_point = plane_line_intersection((0, 0, 1), (0, 0, 0), (0, 0, -1), (0, 0, 1))
    assert np.allclose(intersection_point, (0, 0, 0))

    intersection_point = plane_line_intersection((0, 0, -1), (0, 0, 1), (0, 0, -1), (2, 0, 3))
    assert np.allclose(intersection_point, (1, 0, 1))

    intersection_point = plane_line_intersection((0, 0, 1), (0, 0, 1), (0, 0, -1), (2, 0, 3))
    assert np.allclose(intersection_point, (1, 0, 1))

    intersection_point = plane_line_intersection((0, 0, 1), (0, 11, 1), (0, 22, -1), (2, 22, 3))
    assert np.allclose(intersection_point, (1, 22, 1))

    intersection_point = plane_line_intersection((0, 0, 1), (0, 0, 1), (0, 0, 1), (5, 0, 3))
    assert np.allclose(intersection_point, (0, 0, 1))

    intersection_point = plane_line_intersection((0, 0, 1), (0, 0, 2), (0, 0, 1), (0, 0, 1))
    assert intersection_point is None
