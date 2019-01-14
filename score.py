import numpy as np
from math import pi


def normalize_quaternions(q, eps=1e-12):

    """ Normalizing quaternion(s) to unit norm. """

    if type(q) is np.ndarray:
        norm = np.maximum(np.linalg.norm(q, axis=1), eps)
        return q / norm[:, np.newaxis]
    else:
        q = np.array(q)
        norm = np.maximum(np.linalg.norm(q), eps)
        return q / norm


def batch_phi(q):

    """ Calculating roll for batch of quaternions. """

    sinr_cosp = 2.0 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
    cosr_cosp = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    return np.expand_dims(roll, 1)


def batch_theta(q):

    """ Calculating pitch for batch of quaternions. """

    sinp = 2.0 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1])
    pitch = np.ones(sinp.shape) * pi/2 * np.sign(sinp)
    pitch[np.abs(sinp)<1] = np.arcsin(sinp[np.abs(sinp)<1])

    return np.expand_dims(pitch, 1)


def batch_psi(q):

    """ Calculating yaw for batch of quaternions. """

    siny_cosp = 2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2])
    cosy_cosp = 1.0 - 2.0 * (q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.expand_dims(yaw, 1)


def convert_to_euler(q):

    """ Converting batch of quaternions to euler angles. """

    q = normalize_quaternions(q)

    pitch = batch_phi(q)
    roll = batch_theta(q)
    yaw = batch_psi(q)

    return np.hstack([pitch, roll, yaw])


def normalized_difference(angles_a, angles_b, use_radians=True):

    """ Calculating angular difference, accounting for periodicity. """

    if use_radians:
        period = 2 * pi
    else:
        period = 360

    diff = angles_a - angles_b

    sign = np.sign(diff)
    sign[sign == 0] = 1
    diff = diff % (sign * period)
    idx = abs(diff) > period / 2
    diff[idx] -= sign[idx] * period
    return diff


def np_array_with_batch_dim(pose):

    """ Converting inputs to ndarrays with batch dimension. """

    pose = np.array(pose)
    if pose.ndim == 1:
        pose = np.expand_dims(pose, 0)
    return pose


def compute(prediction, label):

    """ Calculating evaluation metric for the competition. """

    prediction = np_array_with_batch_dim(prediction)
    label = np_array_with_batch_dim(label)

    pred_q = convert_to_euler(prediction[:, :4])
    label_q = convert_to_euler(label[:, :4])

    pred_r = prediction[:, 4:]
    label_r = label[:, 4:]

    # angle error norm
    euler_error = normalized_difference(label_q, pred_q)
    euler_error_2_norm = np.linalg.norm(euler_error, axis=1)

    # distance error norm, normalized with target distance
    target_distances = np.linalg.norm(label_r, axis=1)
    translation_error = label_r - pred_r
    trans_error_2_norm = np.linalg.norm(translation_error)
    translation_error_dist_normalized = trans_error_2_norm / target_distances

    # final score: adding error norm, calulating mean
    scores = euler_error_2_norm + translation_error_dist_normalized

    return scores.mean()


if __name__ == "__main__":

    # debug
    labels = np.random.random([42, 7])
    predictions = np.random.random([42, 7])
    print('Score: {}'.format(compute(predictions, labels)))


