try:
    from local_settings import environment
except ImportError:
    environment = 'prod'

import json
import os


# SETTINGS
submission_prediction_keys = {'filename', 'q', 'r'}

# ENV DEPENDENT SETUP
if environment == 'prod':
    # we are on Kelvins server
    kelvins_root = '/home/kelvins/uploads/competitions/proba-v-super-resolution/'
    with open(os.path.join(kelvins_root, 'speed_debug', 'test.json'), 'r') as f:
        test_image_list = json.load(f)

elif environment == 'dev':
    # local machine
    with open('/datasets/speed_debug/test.json', 'r') as f:
        test_image_list = json.load(f)
else:
    raise ValueError('\nUnexpected environment {}. '.format(environment) +
                     'Set environment in local_settings.py to \'prod\' or \'dev\'!')


test_image_names = set([x['filename'] for x in test_image_list])

# This return the score for the leaderboard
# if there is an error an exception must be thrown here, but will not be visible to the user. Instead
# the submission will be marked invalid and a generic error communicated to the user via the web page.
# May return a single score or a 2 element tuple. In
# which case the first element is the score and the second the extra_info on the leaderboard
def score(file):
    return 0, 'extra_info'


# The following function (if implemented) will be used instead of score. The difference is that it has access
# to all previous submissions to and thus can score some submission with respect to the previous one.
# This trick was used, for example, in the GTOC9 competition where the score depended on all valid submissions.
def score_submission(submission):
    return 0, 'extra_info'


# This runs immmediately after the upload and validates the easy bits (format size etc.)
# if succesfull (no exception) score will be run later (by celery)
# otherwise the text of the exception is shown on the web site (the user sees it) TEST IT PROPERLY
def validate(file):

    if environment == 'prod':
        if file.size > 15 * (1 << 20):
            raise ValueError('File size too big, maximum is 15 MB.')

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

    # from code import interact
    # interact(local=locals())

    if checked_images != test_image_names:
        missing_image_names = list(test_image_names - checked_images)
        missing_image_names.sort()
        raise ValueError('The pose for the following images is missing: {}.'.format(missing_image_names))
