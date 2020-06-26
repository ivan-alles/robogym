#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import numpy as np
import warnings

def plane_line_intersection(l0, l, p0, n):
    """
    Intersection of a plane and a line.

    # TODO(ia): also support passing a line and point equations.

    See https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection#Algebraic_form.
    :param l0: a point on the line
    :param l: another point on the line
    :param n: plane normal
    :param p0: a point on the plane.
    :return: intersection point [x, y, z]
    """
    l0 = np.array(l0).squeeze()
    l = np.array(l).squeeze()
    p0 = np.array(p0).squeeze()
    n = np.array(n).squeeze()

    l_dot_n = np.dot(l, n)
    if abs(l_dot_n) < 1e-5:
        # Line and plane are parallel
        return None
    d = np.dot(p0 - l0, n) / l_dot_n
    if abs(d) < 1e-5:
        # Plane contains the line
        return None
    ip = d * l + l0
    return ip

def line_by_points(p0, p1):
    """
        Compute a normalized equation of the straight line passing through two points.
        Line normal is oriented as the y-axis relative to the x-axis:
        straight_line_points((0, 0), (1, 0)) == (0, 1, 0).
        @return
        :param p0: point 1 (x, y)
        :param p1: point 1 (x, y)
        :return: a normalized line equation l0*x + l1*y + l2 = 0.
    """
    p0 = np.array(p0)
    p1 = np.array(p1)

    if np.array_equal(p0, p1):
        raise ValueError("Cannot compute line equation: 2 points are equal")

    e = np.array([p0[1] - p1[1], p1[0] - p0[0]], dtype=np.float32)
    n = np.linalg.norm(e)
    e /= n
    return np.append(e, -e[0] * p0[0] - e[1] * p0[1])

def plane_by_normal_point(n, p):
    """
    Computes a plane equation by given normal vector and a point on the plane.

    :return: a plane equation [nx, ny, nz, d], where (nx, ny, nz) is a unit-normal vector.
    """
    n = np.array(n, dtype=np.float32).squeeze()
    p = np.array(p, dtype=np.float32).squeeze()
    if len(n) != 3:
        raise ValueError("Normal size must be 3")
    if len(p) != 3:
        raise ValueError("Point size must be 3")

    nnorm = np.linalg.norm(n)
    epsilon = 1e-10
    if nnorm < epsilon:
        raise ValueError("Normal vector has zero length")
    e = np.zeros(4, dtype=np.float32)
    e[0:3] = n / nnorm
    e[3] = -np.dot(n, p)
    return e

def line_point_distance(line, point):
    """
    Computes distance between a line and point.
    :param line: a line equation [a, b, c]: ax + by + c = 0.
    :param point: a point [x, y] or an array of any shape of such points.
    :return: distance to point(s).
    """
    line = np.array(line)
    return (line[:2] * point).sum(axis=-1) + line[2]


def intersect_rects(r1, r2):
    """
    Intersect 2 2d rectangles.

    :param r1: 1st rectangle given by [x, y, width, height].
    :param r2: 2nd rectangle.
    :return: intersection of 2 rectangles.
    """
    r1 = np.array(r1).reshape(2, 2)
    r2 = np.array(r2).reshape(2, 2)
    # Convert to point, point representation
    r1[1, :] += r1[0, :]
    r2[1, :] += r2[0, :]
    ir = np.empty((2, 2), dtype=np.float32)
    for d in range(2):
        ir[0, d] = max(r1[0, d], r2[0, d])
        ir[1, d] = min(r1[1, d], r2[1, d])
    # Convert to point, size representation
    ir[1, :] -= ir[0, :]
    return ir.reshape(4)


def normalize_angle(angle, period = np.pi * 2, start = None):
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
    :axis: Axis or axes along which the means are computed.
    :return: mean.
    """

    s = np.sin(angles)
    c = np.cos(angles)
    m = np.arctan2(s.sum(axis=axis), c.sum(axis=axis))

    return m

def line_intersection(l1, l2, epsilon=1e-10):
    """
    Compute an intersection point of two 2d lines.
    See https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    :param l1: line equation 1.
    :param l2: line equation 2.
    :return: a point [x, y] or none.
    """
    c = np.cross(l1, l2)

    if np.abs(c[2]) < epsilon:
        return None

    return (c / c[2])[:2]
