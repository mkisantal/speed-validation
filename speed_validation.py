try:
    from local_settings import environment
except ImportError:
    environment = 'prod'

import json
import os
import numpy as np
from math import pi


# SETTINGS
submission_partition_keys = {'test', 'tron'}
submission_prediction_keys = {'filename', 'q', 'r'}
max_filesize = 2  # MB

# ENV DEPENDENT SETUP
if environment == 'prod':
    # we are on Kelvins server
    kelvins_root = '/srv/kelvins.esa.int/code-git/uploads/competitions/satellite-pose-estimation-challenge'
    with open(os.path.join(kelvins_root, 'test.json'), 'r') as f:
        test_image_list = json.load(f)
    with open(os.path.join(kelvins_root, 'tron.json'), 'r') as f:
        tron_image_list = json.load(f)
    with open(os.path.join(kelvins_root, 'test_labels.json'), 'r') as f:
        test_labels = json.load(f)
    with open(os.path.join(kelvins_root, 'tron_labels.json'), 'r') as f:
        tron_labels = json.load(f)
    root = kelvins_root

elif environment == 'dev':
    # local machine
    with open('/datasets/speed_debug/test.json', 'r') as f:
        test_image_list = json.load(f)
    with open('/datasets/speed_debug/tron.json', 'r') as f:
        tron_image_list = json.load(f)
    with open('/datasets/speed_debug_TEST_LABELS/test_labels.json', 'r') as f:
        test_labels = json.load(f)
    with open('/datasets/speed_debug_TEST_LABELS/tron_labels.json', 'r') as f:
        tron_labels = json.load(f)
    root = ''
else:
    raise ValueError('\nUnexpected environment {}. '.format(environment) +
                     'Set environment in local_settings.py to \'prod\' or \'dev\'!')

test_and_tron_image_list = test_image_list + tron_image_list
test_and_tron_image_names = set([x['filename'] for x in test_and_tron_image_list])


# SCORING AND VALIDATION
def score(file):

    """ Call scoring function, log exceptions, re-raise error. """

    try:
        scr, inf = _score(file)
        return scr, inf
    except Exception as e:
        with open(os.path.join(root, 'score_error_log.log'), 'a+') as log_file:
            print(str(e), file=log_file)
        raise e


def _score(file):
    predictions = json.load(file)

    test_estimates = predictions['test']
    tron_estimates = predictions['tron']

    # sort everything, just in case
    for estimate_list in [test_estimates, tron_estimates, test_labels, tron_labels]:
        estimate_list.sort(key=lambda k: k['filename'])

    test_estimates_pose = [x['q'] + x['r'] for x in test_estimates]
    test_labels_pose = [x['q_vbs2tango'] + x['r_Vo2To_vbs_true'] for x in test_labels]
    test_score = compute(test_estimates_pose, test_labels_pose)

    tron_estimates_pose = [x['q'] + x['r'] for x in tron_estimates]
    tron_labels_pose = [x['q_vbs2tango'] + x['r_Vo2To_vbs_true'] for x in tron_labels]
    tron_score = compute(tron_estimates_pose, tron_labels_pose)

    # for estimate, ground_truth in zip(test_estimates, test_pose_labels):
    #     if estimate['filename'] != ground_truth['filename']:
    #         raise ValueError('Something got really messed up, inconsistent file names:' +
    #                          '\'{}\' \'{}\''.format(estimate['filename'], ground_truth['filename']))

    return test_score, str(tron_score)


def validate(file):

    if environment == 'prod':
        if file.size > max_filesize * (1 << 20):
            raise ValueError('File size too big, maximum is {} MB.'.format(max_filesize))

    try:
        predictions = json.load(file)
    except json.decoder.JSONDecodeError:
        raise ValueError('Json decoding error. The uploaded .json file is invalid.')

    # check if both test and tron estimates are in submission file.
    try:
        if set(predictions.keys()) != submission_partition_keys:
            raise ValueError('Pose estimates required for both test and tron images.\n' +
                             'Expected keys {} in submission file, '.format(submission_partition_keys) +
                             'but only key(s) {} was found. '.format(predictions.keys()))
    except AttributeError:
        raise AttributeError('Submission file should contain dict with keys {}.'.format(submission_partition_keys) +
                             ' Parsed file contained {} instead.'.format(type(predictions)))

    test_poses = predictions['test']
    tron_poses = predictions['tron']

    checked_images = set()
    # check each submitted pose
    for poses in [test_poses, tron_poses]:
        for i, prediction in enumerate(poses):

            # check keys
            if set(prediction.keys()) != submission_prediction_keys:
                raise ValueError('Expected keys {} in submission file, instead got keys:'.format(submission_prediction_keys,
                                                                                                 prediction.keys))
            # check filename
            if prediction['filename'] not in test_and_tron_image_names:
                raise ValueError('Image filename \'{}\' not in test image names.'.format(prediction['filename']))

            # check pose variable sizes
            if len(prediction['q']) != 4:
                raise ValueError('Expected list with 4 variables for quaternion,' +
                                 ' got {} instead:'.format(len(prediction['q'])) +
                                 '\n[{}]'.format(prediction['q']))
            if len(prediction['r']) != 3:
                raise ValueError('Expected list with 3 variables for translation vector,' +
                                 ' got {} instead:'.format(len(prediction['q'])) +
                                 '\n[{}]'.format(prediction['q']))

            checked_images.add(prediction['filename'])

    if checked_images != test_and_tron_image_names:
        missing_image_names = list(test_and_tron_image_names - checked_images)
        missing_image_names.sort()
        raise ValueError('The pose for the following images is missing: {}.'.format(missing_image_names))


# SCORING UTILS

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
