#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import numpy as np


def normalize_angle(angle, period = np.pi * 2, start=None):
    """
    Transforms the angle to the value in range [start, start + period].

    :param angle: angle
    :param period: period
    :param start: minimal value of the resulting angle. If None, is set to -period/2.
    :return: converted angle
    """

    if start is None:
        start = -period / 2

    return (angle - start) % period + start


def mean_of_angles(angles, axis=None):
    """
    Compute mean of angular values as described in https://en.wikipedia.org/wiki/Mean_of_circular_quantities.

    :param angles: an array of angles.
    :param axis: Axis or axes along which the means are computed.
    :return: mean.
    """
    s = np.sin(angles)
    c = np.cos(angles)
    m = np.arctan2(s.sum(axis=axis), c.sum(axis=axis))
    return m


def line_by_points(p0, p1, epsilon=1e-10):
    """
        Compute the equation of the straight line passing through two points.
        Line normal has a length of 1 is oriented as the y-axis relative to the x-axis.
        :param p0: point 1 (x, y)
        :param p1: point 1 (x, y)
        :param epsilon: comparison precision.
        :return: the line equation (a, b, c): a*x + b*y + c = 0 or None if points are equal.
    """
    p0 = np.array(p0)
    p1 = np.array(p1)

    if np.allclose(p0, p1, atol=epsilon):
        return None

    e = np.array([p0[1] - p1[1], p1[0] - p0[0]])
    n = np.linalg.norm(e)
    e = e / n
    return np.append(e, -e[0] * p0[0] - e[1] * p0[1])


def line_point_distance(line, point):
    """
    Computes distance between a line and point.
    :param line: a line equation [a, b, c]: ax + by + c = 0.
    :param point: a point [x, y] or an array of any shape of such points.
    :return: distance to point(s).
    """
    line = np.array(line)
    return (line[:2] * point).sum(axis=-1) + line[2]


def line_intersection(l1, l2, epsilon=1e-10):
    """
    Compute an intersection point of two 2d lines.
    See https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    :param l1: line equation 1.
    :param l2: line equation 2.
    :param epsilon: comparison precision.
    :return: a point [x, y] or None if lines are parallel or equal.
    """
    c = np.cross(l1, l2)

    if np.abs(c[2]) < epsilon:
        return None

    return (c / c[2])[:2]


def plane_by_normal_point(n, p, epsilon=1e-10):
    """
    Computes a plane equation by given normal vector and a point on the plane.
    :param n: a normal to the plane.
    :param p: a point on the plane.
    :param epsilon: comparison precision.
    :return: a plane equation [nx, ny, nz, d], where (nx, ny, nz) is a unit-normal vector.
    """
    n = np.array(n).squeeze()
    p = np.array(p).squeeze()
    if len(n) != 3:
        raise ValueError("Normal size must be 3")
    if len(p) != 3:
        raise ValueError("Point size must be 3")

    n_norm = np.linalg.norm(n)

    if n_norm < epsilon:
        raise ValueError("Normal vector has zero length")
    plane_eq = np.zeros(4)
    plane_eq[:3] = n / n_norm
    plane_eq[3] = -np.dot(plane_eq[:3], p)
    return plane_eq


def plane_line_intersection(plane_n, plane_p0, line_p0, line_p1, epsilon=1e-10):
    """
    Intersection of a plane and a line.

    See https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection#Algebraic_form.
    :param line_p0: a point on the line.
    :param line_p1: another point on the line.
    :param plane_n: plane normal.
    :param plane_p0: a point on the plane.
    :param epsilon: comparison precision.
    :return: intersection point (x, y, z).
    """
    line_p0 = np.array(line_p0).squeeze()
    line_p1 = np.array(line_p1).squeeze()
    plane_p0 = np.array(plane_p0).squeeze()
    plane_n = np.array(plane_n).squeeze()

    line_l = line_p1 - line_p0

    l_dot_n = np.dot(line_l, plane_n)
    if abs(l_dot_n) < epsilon:
        # Line and plane are parallel
        return None
    d = np.dot(plane_p0 - line_p0, plane_n) / l_dot_n
    if abs(d) < epsilon:
        # Plane contains the line
        return line_p0
    intersection_point = d * line_l + line_p0
    return intersection_point








