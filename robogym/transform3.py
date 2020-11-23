#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

""" 3d transformations. """

import numpy as np
from numpy.linalg import norm, inv
from scipy.spatial.transform import Rotation


class Transform3:
    """ A transformation for in 3d space.

        Can be use for rigid, affine, projective and other transformations or a combination of all the above.
        Stores the transformation as 4x4 matrix (actually np.ndarray) internally.
        The class is designed for simplicity and convenience of usage, not for efficiency.

        The interface somewhat resembles numpy ndarray, to make it easier to port from typical numpy code
        with a lot of dot() and inv() calls.
        Usage of * operator is deliberately avoided as it resembles numpy element-wise mulitplication.
    """

    EPSILON = 1e-10
    DTYPE = np.float64

    def __init__(self, *args):
        """
        Creates a new transformation from various parameters:
        :param args:
        * One argument:
           * A 4x4 matrix is directly copied to the internal matrix.
           * A 3x3 (e.g. rotation or intrinsics) or 3x4 matrix (e.g. rotation and translation). It is copied to the
             upper left corner of the internal matrix .
           * A vector of size 6 is treated as 6-DOF pose (tx, ty, tz, rx, ry, rz).
           * A vector of size 3 is treated as a translation.
           * A scalar or an array with 1 element will create a uniform-scale matrix.

        * Two arguments
           First argument is a translation vector of size 3.
           Second argument can be:
           * 3x3 matrix which is copied to the internal matrix (e.g. rotation matrix).
           * rotation vector of size 3.

        In case an incomplete (less than 4x4) matrix is specified, internal matrix is first set to identity,
        than the partial matrices overwrite it.
        """

        if len(args) == 0:
            self._m = np.identity(4, dtype=Transform3.DTYPE)
        elif len(args) == 1:
            arg = args[0]
            arg_type = type(arg)
            if arg_type == Transform3:
                self._m = np.copy(arg._m)
            elif arg_type == str:
                self._m = Transform3.from_str(arg).m
            else:
                self._init_from_one_array(arg)
        elif len(args) == 2:
            self._init_from_two_arrays(args[0], args[1])

        if not hasattr(self, '_m'):
            raise ValueError('Cannot create Transform3 with given arguments: ' + str(args))

    @staticmethod
    def from_axis_angle(axis, angle=None):
        """
        Create a rotation around from axis by an angle centered at origin.

        Use @staticmethod here and later because this is a kind of constructor.
        :param axis: axis of rotation.
        :param angle: angle of rotation. If None, the angle equals the norm of the axis.
        :return: transform.
        """
        return Transform3.from_t_axis_angle(np.array([0, 0, 0], Transform3.DTYPE), axis, angle)

    @staticmethod
    def from_center_axis_angle(center, axis, angle=None):
        """
        Create a rotation around from axis by an angle centered at a given center.

        :param center: center of rotation.
        :param axis: axis of rotation.
        :param angle: angle of rotation. If None, the angle equals the norm of the axis.
        :return: transform.
        """
        tr = Transform3.from_axis_angle(axis, angle)
        c = center.squeeze()
        tr._m[0:3, 3:] = (c - tr.r.dot(c)).reshape(3, 1)
        return tr

    @staticmethod
    def from_t_axis_angle(t, axis, angle):
        """
        Create a transform from translation, axis and angle.

        :param t: translation.
        :param axis: axis of rotation
        :param angle: angle of rotation. If None, the angle equals the norm of the axis.
        :return: transform.
        """
        axis = np.array(axis, dtype=Transform3.DTYPE).squeeze()
        t = np.array(t, dtype=Transform3.DTYPE).squeeze()
        if angle is not None:
            n = norm(axis)
            if n < Transform3.EPSILON:
                if angle > Transform3.EPSILON:
                    raise ValueError('Cannot rotate around a zero-length axis')
            else:
                axis /= n
            axis *= angle

        r = Rotation.from_rotvec(axis).as_matrix()
        return Transform3.from_t_r(t, r)

    @staticmethod
    def from_t(t):
        """
        Create a transform from a rotation vector.

        :param t: translation vector of size 3.
        :return: transform.
        """
        r = np.identity(3, Transform3.DTYPE)
        return Transform3.from_t_r(t, r)

    @staticmethod
    def from_pose6(pose):
        """
        Create a transform from 6 DOF pose.

        :param pose: 6 DOF pose (tx, ty, tz, rx, ry, rz)
        :return: transform.
        """
        pose = np.array(pose).squeeze()
        return Transform3.from_t_axis_angle(pose[0:3], pose[3:6], None)

    @staticmethod
    def from_t_r(t, r):
        """
        Create a transform from a translation vector and 3x3 rotation matrix.

        :param t: translation vector of size 3.
        :param r: rotation matrix
        :return: transform.
        """
        r = np.array(r, Transform3.DTYPE).squeeze()
        t = np.array(t, Transform3.DTYPE).squeeze()
        tr = Transform3()
        tr._m[0:3, 0:3] = r
        tr._m[0:3, 3:] = t.reshape(3, 1)
        return tr

    @staticmethod
    def from_scale(scale):
        """
        Create a a scale transform.

        :param scale: if it is a scalar, a uniform scale transform is created.
        Otherwise it must be a vector of size 3 [sx, sy, sz].
        :return: transform.
        """
        scale = np.array(scale).squeeze()
        if scale.size == 1:
            s = scale.reshape(1)[0]
            scale = np.array([s, s, s])
        scale = np.append(scale, 1)
        mat = np.diag(scale).astype(Transform3.DTYPE)
        return Transform3(mat)

    @staticmethod
    def from_str(s, sep=' '):
        return Transform3(np.fromstring(s, sep=sep, dtype=Transform3.DTYPE))

    def _init_from_one_array(self, a):
        a = np.array(a, Transform3.DTYPE).squeeze()
        if a.shape == (16,):
            a = a.reshape((4, 4))
        elif a.shape == (12,):
            a = a.reshape((3, 4))
        elif a.shape == (9,):
            a = a.reshape((3, 3))

        if a.shape == (4, 4):
            self._m = a.astype(Transform3.DTYPE, copy=True)
        elif a.shape == (3, 3) or a.shape == (3, 4):
            self._m = np.identity(4, Transform3.DTYPE)
            self._m[0:a.shape[0], 0:a.shape[1]] = a
        elif a.shape == (7, ):
            self._m = Transform3.from_t_axis_angle(a[:3], a[3:6], a[6])._m
        elif a.shape == (6, ):
            self._m = Transform3.from_pose6(a)._m
        elif a.shape == (3, ):
            self._m = Transform3.from_t(a)._m
        elif a.size == 1:
            # a scalar - create scale transform
            self._m = Transform3.from_scale(a)._m

    def _init_from_two_arrays(self, a1, a2):
        # Remove 1-dimensions. Vectors nx1 become single-dimensional
        a1 = np.array(a1, Transform3.DTYPE).squeeze()
        a2 = np.array(a2, Transform3.DTYPE).squeeze()

        if a2.shape == (3, ):
            self._m = Transform3.from_t_axis_angle(a1, a2, None)._m
        else:
            self._m = Transform3.from_t_r(a1, a2)._m

    def is_rotation(self):
        """
        Check if self.r is a rotation matrix. In other words,  it preserves angles and is not a symmetry transform.
        The transformation part does not matter.
        :return: True if self.r is a rotation matrix.
        """

        s = dot(self.r, self.r.T)
        if not np.allclose(s, np.identity(self.r.shape[0], self.r.dtype), atol=Transform3.EPSILON):
            return False

        d = np.linalg.det(self.r)

        if abs(d - 1) > Transform3.EPSILON:
            return False

        return True

    @property
    def m(self):
        """
        :return: (a reference to) the internal 4x4 matrix.
        """
        return self._m

    @property
    def r(self):
        """
        :return: (a reference to) 3x3 rotaion component of the matrix
        """
        return self._m[0:3, 0:3]

    @r.setter
    def r(self, value):
        self._m[0:3, 0:3] = np.array(value, Transform3.DTYPE).squeeze()

    @property
    def t(self):
        """
        :return: (a reference to) translation component of the matrix of shape (3,).
        """
        return self._m[0:3, 3]

    @t.setter
    def t(self, value):
        self._m[0:3, 3] = np.array(value, Transform3.DTYPE).reshape(3)

    @property
    def ax(self):
        """
        :return: (a reference to) the axis x of the coordinate frame of shape (3,).
        """
        return self._m[0:3, 0]

    @ax.setter
    def ax(self, value):
        self._m[0:3, 0] = np.array(value, Transform3.DTYPE).reshape(3)

    @property
    def ay(self):
        """
        :return: (a reference to) the axis y of the coordinate frame of shape (3,).
        """
        return self._m[0:3, 1]

    @ay.setter
    def ay(self, value):
        self._m[0:3, 1] = np.array(value, Transform3.DTYPE).reshape(3)

    @property
    def az(self):
        """
        :return: (a reference to) the axis z of the coordinate frame of shape (3,).
        """
        return self._m[0:3, 2]

    @az.setter
    def az(self, value):
        self._m[0:3, 2] = np.array(value, Transform3.DTYPE).reshape(3)

    def __getitem__(self, idx):
        return self._m.__getitem__(idx)

    def __setitem__(self, idx, val):
        return self._m.__setitem__(idx, val)

    def rvec(self):
        """
        :return: rotaton vector (axis of rotation with the norm equals to the angle of rotation).
        """
        return Rotation.from_matrix(self.r).as_rotvec()

    def pose6(self):
        """
        Convert to a 6 DOF (tx, ty, tz, rx, ry, rz) representation.
        :return:
        """
        rv = self.rvec()
        t = self.t.squeeze()
        p6 = np.concatenate([t, rv])
        return p6

    def t_axis_angle(self):
        """
        Convert to a t, axis, angle representation.
        :return: t, axis (of unit length), angle.
        """
        rv = self.rvec()
        angle = np.linalg.norm(rv)
        axis = rv / angle if angle > 0 else np.array([1, 0, 0], dtype=Transform3.DTYPE)
        t = self.t.squeeze()
        return t, axis, angle

    def inv(self):
        """
        Inverse the transform.
        :return: inverted transform.
        """
        return Transform3(inv(self._m))

    def transp(self):
        """
        Transpose the transform.

        :return: transform transpose.
        """

        return Transform3(self._m.T)

    def dot(self, rhs, transpose=False):
        """
        Compute dot(_m, rhs), input will be adapted if necessary to match the matrix size.

        :param rhs: right hand side of the multiplication. Can be:
        * Another transform. In this case the returned value is a new transform.

        * A vector or array vectors of size 3xN or 4xN. A subcase of it is a matrix 4x4.
        The required input layout is 1d vector or a 2d array of column vectors of size 3 or 4.
        If the size of the vectors is 3, they will be augmented with a row of ones,
           otherwise they will be treated as a homogeneous vectors.

        :param transpose if true, the input is transposed before transformation, and the output is transposed back.

        :return: The transformed rhs with the original layout.

        """
        rhs_type = type(rhs)
        if rhs_type == Transform3:
            rhs = rhs._m

        r = self._dot_vec(rhs, transpose)

        if rhs_type == Transform3:
            r = Transform3(r)

        return r

    def _dot_vec(self, input, transpose):
        # Developer notes: we could have added auto-transposing of inputs of sizes Mx3 or Mx4,
        # but this leads to ambiguities with sizes like 3x3, 3x4, 4x3, 4x4 and probably does more harm than good.
        # Just think about an algo returning an array of row-vectors of variable size. This array cannot be safely
        # passed to this method, because it will be sometimes transposed, sometimes not.
        # So, it's more safe to let the user transform the data.

        inp = np.array(input, Transform3.DTYPE)

        if inp.ndim < 1 or inp.ndim > 2:
            raise ValueError('Input dimension must be 1 or 2')

        input_adapter, inp = MatrixAdapter.create(inp, transpose)

        M = inp.shape[0]
        N = inp.shape[1]

        if M not in range(3, 5):
            raise ValueError('Input vectors must have size 3 or 4')

        is_augmented = False
        if M == 3:
            # Augment with ones
            # We could have done a it more efficiently by multiplying by r and adding t,
            # if we were sure that the last row of our matrix is  [0, 0, 0, 1].
            inp = np.append(inp, np.full((1, N), 1, Transform3.DTYPE), axis=0)
            is_augmented = True

        # Do the transform
        out = self._m.dot(inp)

        # Convert layout to the original one
        if is_augmented:
            out = out[0:-1, :]
        out = input_adapter.revert(out)

        return out

    def project(self, rhs, transpose=False):
        """
        Project 3d points to 2d using perspective division by z-coordinate.

        :param rhs: 3xN or 4xN array.
        :return:
        """

        input_adapter, rhs = MatrixAdapter.create(rhs, transpose)

        # Now we have a Mx3 or Mx4 array
        p = self.dot(rhs)
        xy = p[0:2, :]
        z = p[2:3, :]
        p = np.divide(xy, z)
        p = input_adapter.revert(p)

        return p

    def unproject(self, rhs, transpose=False):
        """
        Unproject 2d points to 3d rays.

        :param rhs: 2xN array or 2-vector.
        :return: 3d points with z-coordinate set to 1.
        """

        input_adapter, rhs = MatrixAdapter.create(rhs, transpose)

        # Augment with z=1. We do not need w for 2d points.
        rhs = np.append(rhs, np.full((1, rhs.shape[1]), 1, Transform3.DTYPE), axis=0)

        # Use 3x3 'rotation' part.
        u = self.r.dot(rhs)
        u = input_adapter.revert(u)
        return u

    def unproject_on_plane(self, rhs, plane_eq, transpose=False):
        """
        Unproject 2d points to 3d points on plane.

        :param rhs: 2xN array or 2-vector.
        :param plane_eq: plane equation in form [nx, ny, nz, d].
        :return: 3d points with z-coordinate set to 1.
        """

        plane_eq = np.array(plane_eq, Transform3.DTYPE).squeeze()

        input_adapter, rhs = MatrixAdapter.create(rhs, transpose)

        # Augment with z=1. We do not need w for 2d points.
        rhs = np.append(rhs, np.full((1, rhs.shape[1]), 1, Transform3.DTYPE), axis=0)

        # Use 3x3 'rotation' part.
        u = self.r.dot(rhs)

        # k = -d / (nx * ux + ny * uy + nz * uz), where uz = 0
        k = -plane_eq[3:4] / np.dot(plane_eq[0:3], u)
        u *= k

        u = input_adapter.revert(u)
        return u

    def __str__(self):
        return str(self._m)

    def str_pose6(self, format='{} {} {} {} {} {}'):
        """
        Convert to a 6 DOF string representation.
        :format a string with formats for each field.
        :return: a string 'tx ty tz rx ry rz'.
        """
        p6 = self.pose6()
        return format.format(*p6)

    def str_t_axis_angle(self, format='{} {} {} {} {} {} {}'):
        """
        Convert to a t, axis, angle representation.
        :format a string with formats for each field.
        :return: a string 'tx ty tz rx ry rz a'.
        """
        r = np.hstack(self.t_axis_angle())
        return format.format(*r)

    def human_readable_list(self):
        """
        Convert to the most human-readable list for serialization:
        - If it is a rotation matrix, this is t, axis, angle
        - otherwise a list of lists [4x4].
        """
        if self.is_rotation():
            m = np.hstack(self.t_axis_angle())
        else:
            m = self._m

        return m.tolist()


def dot(*args):
    """
    Transform using chain of transforms.

    :param args: a list of transforms or matrices 4x4. The last argument can also be an array of vectors.
    :return: last argument transformed by the previous arguments.
    If the last argument is a transform, a new transform will be returned, as in chain().
    If the last argument is an array, it will be transformed using the given transforms.
    """
    if len(args) < 2:
        raise ValueError('Need at least 2 arguments')

    tr_list = list(args[:-1])
    rhs = args[-1:][0]

    if len(tr_list) > 1:
        tr = chain(*tr_list)
    else:
        tr = Transform3(tr_list[0])

    return tr.dot(rhs)


def chain(*args):
    """
    Chain of transforms.

    :param args: a list of transforms or 4x4 matrices.
    :return: a new transform representing a chain of given transforms.
    """

    a_list = []
    for t in args:
        type_t = type(t)
        if type_t == Transform3:
            a_list.append(t._m)
        elif type_t == np.ndarray:
            if t.shape != (4, 4):
                raise ValueError('Wrong size of matrix')
            a_list.append(t)
        else:
            raise ValueError('Unexpected argument type ', type_t)

    ch = np.linalg.multi_dot(a_list)
    return Transform3(ch)


class MatrixAdapter:
    """
    A helper class, converts the input matrix to a column-vector form if needed
    and makes a reverse transform on the output.
    """

    @staticmethod
    def create(input, transpose):
        adapter = MatrixAdapter()
        input = np.array(input, dtype=Transform3.DTYPE)
        adapter._transpose = transpose
        adapter._ndim = input.ndim
        if adapter._transpose:
            input = input.T
        if adapter._ndim == 1:
            input = input.reshape(-1, 1)
        return adapter, input

    def revert(self, output):
        if self._ndim == 1:
            output = output.squeeze()
        if self._transpose:
            output = output.T
        return output
