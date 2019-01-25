import speed_validation
import numpy as np
from math import sin, cos, pi, sqrt, radians


""" Tests for functions of the score computation. """


def test_np_array_with_batch_dim():

    test_array1 = []
    test_array2 = np.ones([7, 4])

    # is it two dimensional?
    assert speed_validation.np_array_with_batch_dim(test_array1).ndim == 2
    assert speed_validation.np_array_with_batch_dim(test_array1).ndim == 2

    # do we get a numpy array?
    assert isinstance(speed_validation.np_array_with_batch_dim(test_array1), np.ndarray)
    assert isinstance(speed_validation.np_array_with_batch_dim(test_array2), np.ndarray)

    # are dimensions preserved?
    assert len(test_array1) == speed_validation.np_array_with_batch_dim(test_array1).shape[1]
    assert test_array2.shape == speed_validation.np_array_with_batch_dim(test_array2).shape
    return


def test_normalize_quaternion():

    test_quaternion_1 = [0, 0, 0, 0]
    test_quaternion_2 = [1, 0, 0, 0]
    test_quaternion_3 = [2, 0, 0, 0]
    test_quaternion_4 = [sqrt(2)/2, sqrt(2)/2, 0, 0]
    test_quaternion_5 = [2, 2, 0, 0]
    test_quaternion_array = np.zeros([3, 4])

    assert np.all(speed_validation.normalize_quaternions(test_quaternion_1) == np.array([0, 0, 0, 0]))
    assert np.all(speed_validation.normalize_quaternions(test_quaternion_2) == np.array(test_quaternion_2))
    assert np.all(speed_validation.normalize_quaternions(test_quaternion_3) == np.array(test_quaternion_2))
    assert np.all(speed_validation.normalize_quaternions(test_quaternion_4) == np.array(test_quaternion_4))
    assert np.allclose(speed_validation.normalize_quaternions(test_quaternion_5), np.array(test_quaternion_4))
    assert np.all(speed_validation.normalize_quaternions(test_quaternion_array) == np.zeros([3, 4]))

    return


def is_close(a, b, eps=1e-12):
    return abs(a-b) < eps


def test_quat_angle():

    # identical quaternions
    quat_a1 = np.array([[1, 0, 0, 0]])
    quat_b1 = np.array([[1, 0, 0, 0]])
    assert speed_validation.quat_angle(quat_a1, quat_b1) == 0.0

    # q = -q identity
    quat_a2 = np.array([[1, 1, 1, 1]])
    quat_b2 = np.array([[-1, -1, -1, -1]])
    assert speed_validation.quat_angle(quat_a2, quat_b2) == 0.0

    # 90 degree rotation along 3 axes
    quat_a2 = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    quat_b2 = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]])
    assert np.allclose(speed_validation.quat_angle(quat_a2, quat_b2), np.array([pi/2, pi/2, pi/2]))


def test_compute():

    # testing the main score computation function

    # no error
    labels_1 = [[1, 0, 0, 0, .2, .3, 11.1]]
    estimates_1 = labels_1
    assert speed_validation.compute(estimates_1, labels_1) == 0.0

    # TRANSLATION ERRORS
    # 1 meter translation error at 1 meter target distance
    labels_2 = [[1, 0, 0, 0, 0, 0, 1]]
    estimates_2 = [[1, 0, 0, 0, 0, 0, 2]]
    assert speed_validation.compute(estimates_2, labels_2) == 1.0

    # 1 meter translation error  at 1 meter target distance, test averaging
    labels_3 = [[1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1]]
    estimates_3 = [[1, 0, 0, 0, 0, 0, 2],
                   [1, 0, 0, 0, 0, 0, 1]]
    assert speed_validation.compute(estimates_3, labels_3) == 0.5

    # translation error along different axis
    labels_4 = [[1, 0, 0, 0, 0, 0, 1]]
    estimates_4 = [[1, 0, 0, 0, 0, 1, 1]]
    assert speed_validation.compute(estimates_4, labels_4) == 1.0

    # 1 meter error at 10 meter target distance
    labels_6 = [[1, 0, 0, 0, 0, 0, 10]]
    estimates_6 = [[1, 0, 0, 0, 0, 0, 11]]
    assert speed_validation.compute(estimates_6, labels_6) == 0.1

    # ORIENTATION ERRORS
    labels_5 = [1, 0, 0, 0, .1, .2, 10.1]
    labels_5 = [labels_5 for _ in range(3)]

    # 90 degree error along 3 different axis
    estimates_5 = list()
    estimates_5.append(list(rotate_quaternion(labels_5[0][:4], magnitude=90, axis=(1, 0, 0))) + labels_5[0][4:])
    estimates_5.append(list(rotate_quaternion(labels_5[0][:4], magnitude=90, axis=(0, 1, 0))) + labels_5[0][4:])
    estimates_5.append(list(rotate_quaternion(labels_5[0][:4], magnitude=90, axis=(0, 0, 1))) + labels_5[0][4:])
    assert is_close(speed_validation.compute(estimates_5, labels_5), pi/2)

    # pitch over pi/2
    labels_6 = [labels_5[0]]
    estimates_6 = [list(rotate_quaternion(labels_5[0][:4], magnitude=91, axis=(1, 0, 0))) + labels_5[0][4:]]
    assert is_close(speed_validation.compute(estimates_6, labels_6), radians(91))

    # check q == -q identity
    labels_7 = [[1, -1, 1, -1, 0, 0, 1]]
    estimates_7 = [[-1, 1, -1, 1, 0, 0, 1]]
    assert speed_validation.compute(estimates_7, labels_7) == 0

    return


def multiply_quaternion(a, b):
    q1 = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    q2 = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    q3 = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
    q4 = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    return np.array([q1, q2, q3, q4])


def rotate_quaternion(q, magnitude=90, axis=None):
    if axis is None:
        # axis = np.random.random([1, 3])
        axis = (1, 0, 0)
    axis = speed_validation.normalize_quaternions(axis)
    axis = np.squeeze(axis)
    s = sin(magnitude/2*pi/180)
    c = cos(magnitude/2*pi/180)
    q_perturb = np.array([c, axis[0]*s, axis[1]*s, axis[2]*s])
    return multiply_quaternion(q, q_perturb)


def run_all():

    # Run all test functions
    test_np_array_with_batch_dim()
    test_normalize_quaternion()
    test_quat_angle()
    test_compute()
    print('Scoring unit tests passed. ')
