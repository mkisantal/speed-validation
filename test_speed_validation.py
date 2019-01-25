import speed_validation
from math import pi
import test_score_computation


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
    return


def run_score_computation(testcase, expected_score):

    """ Run scoring on submission, test if result is the expected score. """

    sub_file = open(testcase, 'rb')
    speed_validation.validate(sub_file)
    sub_file.seek(0)
    score, info = speed_validation.score(sub_file)
    assert test_score_computation.is_close(score, expected_score, eps=1e-14)
    return


check_if_error_raised('test_cases/invalid.csv', ValueError)
check_if_error_raised('test_cases/wrong_field_type.csv', TypeError)
check_if_error_raised('test_cases/invalid_filename.csv', ValueError)
check_if_error_raised('test_cases/extra_image.csv', ValueError)
check_if_error_raised('test_cases/missing_images.csv', ValueError)
check_if_error_raised('test_cases/wrong_number_of_fields.csv', ValueError)
print('Validation tests successfully completed.\n----------------\n\n')

try:
    test_score_computation.run_all()
    run_score_computation('test_cases_scoring\submission_perfect.csv', 0.0)
    run_score_computation('test_cases_scoring\submission_1_translation.csv', 1.0)
    run_score_computation('test_cases_scoring\submission_90deg_orientation.csv', pi/2)
    print('Score computation tests successfully completed.\n----------------\n\n')
except FileNotFoundError:
    print('In order to run scoring tests, generate the test submissions first by running ' +
          '\'generate_score_test_submissions.py\'!')
