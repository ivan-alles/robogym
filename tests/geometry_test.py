#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import numpy as np

import robogym.geometry as geo

def test_line_by_points():
    # Make sure the line is correctly oriented.
    line = geo.line_by_points((0, 0), (1, 0));
    assert np.allclose((0, 1, 0), line)

def test_line_point_distance():
    line = [1, 0, 2]
    points = [[0, 0], [1, 0], [0, 1], [-2, 1], [-3, -1]]
    assert np.allclose([[2, 3, 2, 0, -1]], geo.line_point_distance(line, points))

def test_normalize_angle():
    assert geo.normalize_angle(0, 360, 0) == 0
    assert geo.normalize_angle(200., 360, 0) == 200
    assert geo.normalize_angle(200., 360) == -160.0
    assert geo.normalize_angle(200, 360, -180) == -160.0

    assert geo.normalize_angle(100, 180) == -80

    assert np.allclose(geo.normalize_angle(np.pi + 0.1, 2 * np.pi), -np.pi + 0.1)
    assert np.allclose(geo.normalize_angle(np.pi + 0.1), -np.pi + 0.1)
    assert np.allclose(geo.normalize_angle(np.pi + 0.1, 2 * np.pi, -np.pi), -np.pi + 0.1)

def test_mean_of_angles():
    m_act = geo.mean_of_angles(np.array([0.0, 0, 0]))
    assert np.allclose(0, m_act)

    m_act = geo.mean_of_angles(np.array([0.1, 0.1, 0.1, 0.1]))
    assert np.allclose(0.1, m_act)

    m_act = geo.mean_of_angles(np.array([0.1, -0.1, 0.1, -0.1]))
    assert np.allclose(0.0, m_act)

    m_act = geo.mean_of_angles(np.array([0, 2*np.pi]))
    assert np.allclose(0.0, m_act)

    m_act = geo.mean_of_angles(np.array([0, 2 * np.pi, -10 * np.pi]))
    assert np.allclose(0.0, m_act)

    m_act = geo.mean_of_angles(np.array([
        [0.1, 0.1 + 2 * np.pi, 0.1 -10 * np.pi],
        [-0.1, -0.1 + 2 * np.pi, -0.1 - 10 * np.pi]
    ]), axis=1)
    assert np.allclose([0.1, -0.1], m_act)


    m_act = geo.mean_of_angles(np.array([
        [0.1, -0.1 + 2 * np.pi, -0.1 - 10 * np.pi, 0.1 + 8 * np.pi],
        [np.pi, 3*np.pi, -5*np.pi, 9*np.pi],
    ]), axis=1)
    assert np.allclose([0, np.pi], m_act)

def test_line_intersection():
    l1 = (0, 1, 0)
    l2 = (-1, -1, 1)
    c = geo.line_intersection(l1, l2)
    assert np.allclose([1, 0], c)



