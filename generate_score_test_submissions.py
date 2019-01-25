
""" This script generates submissions that can be used to test scoring on Kelvins. """

import sys
sys.path.append('../speed-utils/')
from submission import SubmissionWriter
import json
import os
import numpy as np
from test_score_computation import rotate_quaternion


test_case_dir = 'test_cases_scoring'
if not os.path.exists(test_case_dir):
    os.makedirs(test_case_dir)


def identity(input):

    """ No-op. """

    return input


def distance_dependent_translation(r, magnitude=1):

    """ A function to perturb position labels. """

    r = np.array(r)
    length = np.linalg.norm(r)
    random_vector = np.random.random(3)
    error_vector = random_vector/np.linalg.norm(random_vector) * length * magnitude
    return r+error_vector


def export_perturbed_labels(test_labels,
                            real_test_labels,
                            csv_name,
                            q_perturbation_function=identity,
                            r_perturbation_function=identity):

    """ Generate test case with given r and q perturbation. """

    submission = SubmissionWriter()

    for test_set in [test_labels, real_test_labels]:
        for test_label in test_set:

            filename = test_label['filename']
            q = test_label['q_vbs2tango']
            r = test_label['r_Vo2To_vbs_true']

            q2 = q_perturbation_function(q)
            r2 = r_perturbation_function(r)

            submission.append_test(filename,
                                   list(np.squeeze(q2)),
                                   list(np.squeeze(r2)))

    submission.export(out_dir=test_case_dir, suffix=csv_name)
    return


labels_root = '/datasets/speed_debug_TEST_LABELS'

with open(os.path.join(labels_root, 'test_labels.json'), 'r') as f:
    test_label_list = json.load(f)
with open(os.path.join(labels_root, 'real_test_labels.json'), 'r') as f:
    real_test_label_list = json.load(f)
with open(os.path.join(labels_root, 'partial_evaluation_indices.json'), 'r') as f:
    partial_evaluation_indices = json.load(f)


for estimate_list in [test_label_list, real_test_label_list]:
    estimate_list.sort(key=lambda k: k['filename'])


# GENERATING PERTURBED SUBMISSIONS

# perfect solution
export_perturbed_labels(test_label_list, real_test_label_list, 'perfect')

# unit translation error
export_perturbed_labels(test_label_list, real_test_label_list, '1_translation',
                        r_perturbation_function=distance_dependent_translation)

# 90 deg orientation error
export_perturbed_labels(test_label_list, real_test_label_list, '90deg_orientation',
                        q_perturbation_function=lambda q: rotate_quaternion(q, magnitude=90, axis=(0, 1, 0)))






