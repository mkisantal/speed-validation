try:
    from local_settings import environment
except ImportError:
    environment = 'prod'

import json
import os


# SETTINGS
submission_prediction_keys = {'filename', 'q', 'r'}
max_filesize = 2  # MB

# ENV DEPENDENT SETUP
if environment == 'prod':
    # we are on Kelvins server
    kelvins_root = '/srv/kelvins.esa.int/code-git/uploads/competitions/satellite-pose-estimation-challenge'
    with open(os.path.join(kelvins_root, 'test.json'), 'r') as f:
        test_image_list = json.load(f)
    with open(os.path.join(kelvins_root, 'test_labels.json'), 'r') as f:
        test_pose_labels = json.load(f)
    root = kelvins_root

elif environment == 'dev':
    # local machine
    with open('/datasets/speed_debug/test.json', 'r') as f:
        test_image_list = json.load(f)
    with open('/datasets/speed_debug_TEST_LABELS/test_labels.json', 'r') as f:
        test_pose_labels = json.load(f)
    root = ''
else:
    raise ValueError('\nUnexpected environment {}. '.format(environment) +
                     'Set environment in local_settings.py to \'prod\' or \'dev\'!')


test_image_names = set([x['filename'] for x in test_image_list])


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

    # sort, just in case
    predictions.sort(key=lambda k: k['filename'])
    test_pose_labels.sort(key=lambda k: k['filename'])

    for predicted, ground_truth in zip(predictions, test_pose_labels):
        if predicted['filename'] != ground_truth['filename']:
            raise ValueError('Something got really messed up, inconsistent file names:' +
                             '\'{}\' \'{}\''.format(predicted['filename'], ground_truth['filename']))

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
        raise ValueError('The uploaded .json file is invalid.')

    checked_images = set()
    # check each submitted pose
    for i, prediction in enumerate(predictions):

        # check keys
        if set(prediction.keys()) != submission_prediction_keys:
            raise ValueError('Expected keys {} in submission file, instead got keys:'.format(submission_prediction_keys,
                                                                                             prediction.keys))
        # check filename
        if prediction['filename'] not in test_image_names:
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

    if checked_images != test_image_names:
        missing_image_names = list(test_image_names - checked_images)
        missing_image_names.sort()
        raise ValueError('The pose for the following images is missing: {}.'.format(missing_image_names))
