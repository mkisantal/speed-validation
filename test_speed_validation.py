import speed_validation


class UndetectedException(Exception):
    pass


def check_if_error_raised(testcase, expected_error):

    """ Checking if test file raises the error that is expected. """

    sub_file = open(testcase, 'r')
    try:
        speed_validation.validate(sub_file)
        raise UndetectedException('No exception was raised for the test file: {}'.format(testcase))
    except expected_error as e:
        print('[{}] error successfully detected: {}'.format(testcase, e))

    sub_file.close()

# Todo: use django objects for passing the file
# from django.db import models
# from django.db.models.fields.files import FieldFile
# file = FieldFile(None, None, None)
# print(file)
# file.open('submission_debug.json')
# print(file.size)


print('\n\n' + '='*30 + '\nRunning tests with corrupted submission files:\n\n')
check_if_error_raised('test_cases/wrong_filename.json', ValueError)
check_if_error_raised('test_cases/wrong_filename.json', ValueError)
check_if_error_raised('test_cases/wrong_r.json', ValueError)
check_if_error_raised('test_cases/wrong_q.json', ValueError)
check_if_error_raised('test_cases/missing.json', ValueError)
check_if_error_raised('test_cases/invalid.json', ValueError)
print('='*30 + '\nAll error raised as expected.\n\n')

# run scoring script
sub_file = open('submission_debug.json')
score, info = speed_validation.score(sub_file)
print('Ran scoring, score: {}, extra info: {}'.format(score, info))
