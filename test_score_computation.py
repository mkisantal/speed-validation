import speed_validation
import numpy as np
from math import sin, cos, pi, sqrt


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


def test_batch_quat2dcm():

    # test cases cover no rotation, and 90 deg rotation around 3 axes
    test_quaternion_1 = [[1, 0, 0, 0]]
    dcm_1 = np.expand_dims(np.eye(3), 0)

    test_quaternion_2 = [[1, 1, 0, 0]]
    dcm_2 = np.expand_dims([[1, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]], 0).transpose([0, 2, 1])

    test_quaternion_3 = [[1, 0, 1, 0]]
    dcm_3 = np.expand_dims([[0, 0, 1],
                            [0, 1, 0],
                            [-1, 0, 0]], 0).transpose([0, 2, 1])

    test_quaternion_4 = [[1, 0, 0, 1]]
    dcm_4 = np.expand_dims([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 1]], 0).transpose([0, 2, 1])

    assert np.allclose(speed_validation.batch_quat2dcm(test_quaternion_1), dcm_1)
    assert np.allclose(speed_validation.batch_quat2dcm(test_quaternion_2), dcm_2)
    assert np.allclose(speed_validation.batch_quat2dcm(test_quaternion_3), dcm_3)
    assert np.allclose(speed_validation.batch_quat2dcm(test_quaternion_4), dcm_4)

    return


def test_batch_dcm2euler():

    # test cases cover no rotation, and 90 deg rotation around 3 axes
    test_dcm_1 = np.expand_dims(np.eye(3), 0)
    euler_1 = np.array([[0, 0, 0]])

    test_dcm_2 = np.expand_dims([[1, 0, 0],
                                 [0, 0, -1],
                                 [0, 1, 0]], 0).transpose([0, 2, 1])
    euler_2 = np.array([[pi/2, 0, 0]])

    test_dcm_3 = np.expand_dims([[0, 0, 1],
                                 [0, 1, 0],
                                 [-1, 0, 0]], 0).transpose([0, 2, 1])
    euler_3 = np.array([[0, pi/2, 0]])

    test_dcm_4 = np.expand_dims([[0, -1, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]], 0).transpose([0, 2, 1])
    euler_4 = np.array([[0, 0, pi/2]])

    assert np.allclose(speed_validation.batch_dcm2euler(test_dcm_1), euler_1)
    assert np.allclose(speed_validation.batch_dcm2euler(test_dcm_2), euler_2)
    assert np.allclose(speed_validation.batch_dcm2euler(test_dcm_3), euler_3)
    assert np.allclose(speed_validation.batch_dcm2euler(test_dcm_4), euler_4)

    return


def is_close(a, b, eps=1e-12):
    return abs(a-b) < eps


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

    assert speed_validation.compute(estimates_5, labels_5) == pi/2

    return


def multiply_quaternion(a, b):
    q1 = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    q2 = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    q3 = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
    q4 = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    return np.array([q1, q2, q3, q4])


def rotate_quaternion(q, magnitude=1, axis=(1, 0, 0)):
    axis = speed_validation.normalize_quaternions(axis)
    s = sin(magnitude/2*pi/180)
    c = cos(magnitude/2*pi/180)
    q_perturb = np.array([c, axis[0]*s, axis[1]*s, axis[2]*s])
    return multiply_quaternion(q, q_perturb)


if __name__ == "__main__":

    # Run all test functions
    test_np_array_with_batch_dim()
    test_normalize_quaternion()
    test_batch_quat2dcm()
    test_batch_dcm2euler()
    test_compute()
    print('Succesfully ran all scoring function tests. ')
