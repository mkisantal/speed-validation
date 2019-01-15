import speed_validation


class UndetectedException(Exception):
    pass


def check_if_error_raised(testcase, expected_error):

    """ Checking if test file raises the error that is expected. """

    sub_file = open(testcase, 'rb')
    try:
        speed_validation.validate(sub_file)
        raise UndetectedException('No exception was raised for the test file: {}'.format(testcase))
    except expected_error as e:
        print('[{}] error successfully detected:\n\t{}'.format(testcase, e))

    sub_file.close()

# Todo: use django objects for passing the file
# from django.db import models
# from django.db.models.fields.files import FieldFile
# file = FieldFile(None, None, None)
# print(file)
# file.open('submission_debug.json')
# print(file.size)


check_if_error_raised('test_cases/invalid.csv', ValueError)
check_if_error_raised('test_cases/wrong_field_type.csv', TypeError)
check_if_error_raised('test_cases/invalid_filename.csv', ValueError)
check_if_error_raised('test_cases/extra_image.csv', ValueError)
check_if_error_raised('test_cases/missing_images.csv', ValueError)
check_if_error_raised('test_cases/wrong_number_of_fields.csv', ValueError)
print('done.')

# run scoring script
sub_file = open('submission_debug.csv', 'rb')
speed_validation.validate(sub_file)
sub_file.seek(0)
score, info = speed_validation.score(sub_file)
print('Ran scoring, score: {}, extra info: {}'.format(score, info))
