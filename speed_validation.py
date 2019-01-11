try:
    from local_settings import environment
except ImportError:
    environment = 'prod'

import json
import os


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
        test_pose_labels = json.load(f)
    with open(os.path.join(kelvins_root, 'tron_labels.json'), 'r') as f:
        tron_pose_labels = json.load(f)
    root = kelvins_root

elif environment == 'dev':
    # local machine
    with open('/datasets/speed_debug/test.json', 'r') as f:
        test_image_list = json.load(f)
    with open('/datasets/speed_debug/tron.json', 'r') as f:
        tron_image_list = json.load(f)
    with open('/datasets/speed_debug_TEST_LABELS/test_labels.json', 'r') as f:
        test_pose_labels = json.load(f)
    with open('/datasets/speed_debug_TEST_LABELS/tron_labels.json', 'r') as f:
        tron_pose_labels = json.load(f)
    root = ''
else:
    raise ValueError('\nUnexpected environment {}. '.format(environment) +
                     'Set environment in local_settings.py to \'prod\' or \'dev\'!')

test_and_tron_image_list = test_image_list + tron_image_list
test_and_tron_image_names = set([x['filename'] for x in test_and_tron_image_list])


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
    for estimate_list in [test_estimates, tron_estimates, test_pose_labels, tron_pose_labels]:
        estimate_list.sort(key=lambda k: k['filename'])

    for estimate, ground_truth in zip(test_estimates, test_pose_labels):
        if estimate['filename'] != ground_truth['filename']:
            raise ValueError('Something got really messed up, inconsistent file names:' +
                             '\'{}\' \'{}\''.format(estimate['filename'], ground_truth['filename']))

        # call actual scoring function here

    return 0, 'extra_info'

# def score_submission(submission):
#     return 0, 'extra_info'


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
