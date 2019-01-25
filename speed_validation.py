try:
    from local_settings import environment
except ImportError:
    environment = 'prod'

import json
import os
import numpy as np

# DEBUG
counter = 0

# SETTINGS
max_filesize = 2  # MB
partial_evaluation = True

# ENV DEPENDENT SETUP
if environment == 'prod':
    # we are on Kelvins server
    kelvins_root = '/srv/kelvins.esa.int/code-git/uploads/competitions/satellite-pose-estimation-challenge'
    with open(os.path.join(kelvins_root, 'test.json'), 'r') as f:
        test_image_list = json.load(f)
    with open(os.path.join(kelvins_root, 'real_test.json'), 'r') as f:
        real_test_image_list = json.load(f)
    with open(os.path.join(kelvins_root, 'test_labels.json'), 'r') as f:
        test_labels = json.load(f)
    with open(os.path.join(kelvins_root, 'real_test_labels.json'), 'r') as f:
        real_test_labels = json.load(f)
    with open(os.path.join(kelvins_root, 'partial_evaluation_indices.json'), 'r') as f:
        partial_evaluation_indices = json.load(f)
    root = kelvins_root

elif environment == 'dev':
    # local machine
    with open('/datasets/speed_debug/test.json', 'r') as f:
        test_image_list = json.load(f)
    with open('/datasets/speed_debug/real_test.json', 'r') as f:
        real_test_image_list = json.load(f)
    with open('/datasets/speed_debug_TEST_LABELS/test_labels.json', 'r') as f:
        test_labels = json.load(f)
    with open('/datasets/speed_debug_TEST_LABELS/real_test_labels.json', 'r') as f:
        real_test_labels = json.load(f)
    with open('/datasets/speed_debug_TEST_LABELS/partial_evaluation_indices.json', 'r') as f:
        partial_evaluation_indices = json.load(f)
    root = ''
else:
    raise ValueError('\nUnexpected environment {}. '.format(environment) +
                     'Set environment in local_settings.py to \'prod\' or \'dev\'!')

test_and_real_image_list = test_image_list + real_test_image_list
test_and_real_image_names = set([x['filename'] for x in test_and_real_image_list])


def read_csv(file):

    """ Simple csv reading, since csv module fails with the files that django opens in 'rb' mode. """

    lines = file.readlines()
    return [[x for x in line.decode('utf-8').split(',')] for line in lines]


# SCORING AND VALIDATION
def score(file):

    """ Call scoring function, log exceptions, re-raise error. """

    try:
        scr, inf = _score(file, partial_evaluation)
        return scr, inf
    except Exception as e:
        with open(os.path.join(root, 'score_error_log.log'), 'a+') as log_file:
            print(str(e), file=log_file)
        raise e


def _score(file, partial_eval):

    """ Scoring: pairing ground truth with estimates, optionally partial evaluation """

    test_predictions = []
    real_test_predictions = []

    csv_rows = read_csv(file)
    for idx, row in enumerate(csv_rows):
        validate_csv_row(row, idx)
        filename = row[0]
        pose = [float(row[x]) for x in range(1, 8)]
        list_to_append = real_test_predictions if filename.endswith('real.jpg') else test_predictions
        list_to_append.append({'filename': filename, 'pose': pose})

    # sort by filenames
    for estimate_list in [test_predictions, real_test_predictions, test_labels, real_test_labels]:
        estimate_list.sort(key=lambda k: k['filename'])

    # partial evaluation on subset of images
    if partial_eval:
        partial_test_labels = [test_labels[x] for x in partial_evaluation_indices['test']]
        partial_real_labels = [real_test_labels[x] for x in partial_evaluation_indices['real_test']]
        partial_test_predictions = [test_predictions[x] for x in partial_evaluation_indices['test']]
        partial_real_predictions = [real_test_predictions[x] for x in partial_evaluation_indices['real_test']]
    else:
        partial_test_labels = test_labels
        partial_real_labels = real_test_labels
        partial_test_predictions = test_predictions
        partial_real_predictions = real_test_predictions

    test_predictions_pose = [x['pose'] for x in partial_test_predictions]
    real_test_predictions_pose = [x['pose'] for x in partial_real_predictions]

    test_labels_pose = [x['q_vbs2tango'] + x['r_Vo2To_vbs_true'] for x in partial_test_labels]
    real_test_labels_pose = [x['q_vbs2tango'] + x['r_Vo2To_vbs_true'] for x in partial_real_labels]

    test_score = compute(test_predictions_pose, test_labels_pose)
    real_test_score = compute(real_test_predictions_pose, real_test_labels_pose)

    return test_score, str(real_test_score)


def validate_csv_row(row, idx):

    """ Validate row in .csv submission file. """

    if len(row) != 8:
        raise ValueError('[row {}] Row in csv file should contain 8 fields (str filename, 4 floats '.format(idx) +
                         'for orientation quaternion, 3 floats for translation vector), but the following ' +
                         'line contains {} field(s): [{}]'.format(len(row), row))

    # validating fields
    if not row[0].startswith('img'):
        raise ValueError('[row {}] Expected image file name starting with \'img\', got: {}'.format(idx, row[0]))
    for i in range(1, 8):
        try:
            float(row[i])
        except ValueError:
            raise TypeError('[row {}] The following string cannot be converted to float: {}'.format(idx, row[i]))


def validate(file):

    """ Validate .csv submission file. """

    if environment == 'prod':
        if file.size > max_filesize * (1 << 20):
            raise ValueError('File size too big, maximum is {} MB.'.format(max_filesize))

    test_predictions = []
    real_test_predictions = []

    csv_rows = read_csv(file)
    for idx, row in enumerate(csv_rows):
        validate_csv_row(row, idx)
        filename = row[0]
        q = [float(row[x]) for x in [1, 2, 3, 4]]
        r = [float(row[x]) for x in [5, 6, 7]]
        list_to_append = real_test_predictions if filename.endswith('real.jpg') else test_predictions
        list_to_append.append({'filename': filename, 'q': q, 'r': r})

    checked_images = set()
    for predictions in [test_predictions, real_test_predictions]:
        for i, prediction in enumerate(predictions):

            # check filename
            if prediction['filename'] not in test_and_real_image_names:
                raise ValueError('Image filename \'{}\' not in expected filenames.'.format(prediction['filename']))

            checked_images.add(prediction['filename'])

    if checked_images != test_and_real_image_names:
        missing_image_names = list(test_and_real_image_names - checked_images)
        missing_image_names.sort()
        raise ValueError('The pose for the following images is missing: {}.'.format(missing_image_names))

    return


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


def np_array_with_batch_dim(pose):

    """ Converting inputs to ndarrays with batch dimension. """

    pose = np.array(pose)
    if pose.ndim == 1:
        pose = np.expand_dims(pose, 0)
    return pose


def quat_angle(a, b):

    """ Calculating angle between quaternions: z = a * inv(b), |z0| = |dot(a, b)|, theta = 2*arccos(|z0|) """

    # normalize inputs
    x = normalize_quaternions(a)
    y = normalize_quaternions(b)

    batch_size = a.shape[0]
    # dot_prod = torch.bmm(x.view(batch_size, 1, -1), y.view(batch_size, -1, 1)).squeeze()
    dot_prod = np.squeeze(np.matmul(x.reshape(batch_size, 1, -1), y.reshape(batch_size, -1, 1)))
    dot_prod_corrected = np.round(np.clip(dot_prod, -1.0, 1.0), 15)  # to avoid numerical issues
    return 2 * np.arccos(np.abs(dot_prod_corrected))


def compute(prediction, label):

    """ Calculating evaluation metric for the competition. """

    prediction = np_array_with_batch_dim(prediction)
    label = np_array_with_batch_dim(label)

    # orientation error
    pred_q = prediction[:, :4]
    label_q = label[:, :4]
    orientation_difference = quat_angle(pred_q, label_q)

    # translation error
    pred_r = prediction[:, 4:]
    label_r = label[:, 4:]

    # distance error norm, normalized with target distance
    target_distances = np.linalg.norm(label_r, axis=1)
    translation_error = label_r - pred_r
    trans_error_2_norm = np.linalg.norm(translation_error, axis=1)
    translation_error_dist_normalized = trans_error_2_norm / target_distances

    # final score: adding error norm, calculating mean
    scores = orientation_difference + translation_error_dist_normalized
    return scores.mean()
