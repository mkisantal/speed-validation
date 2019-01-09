import sys
sys.path.append('../speed-utils/')
from submission import SubmissionWriter
import json
import os

# load test image list
dataset_root = '/datasets/speed_debug'
with open(os.path.join(dataset_root, 'test.json'), 'r') as f:
    test_set = json.load(f)

submission = SubmissionWriter()


for image in test_set[::-1]:

    filename = image['filename']

    # arbitrary prediction, just to store something.
    q = [1.0, 0.0, 0.0, 0.0]
    r = [10.0, 0.0, 0.0]

    submission.append(filename, q, r)

submission.export()
print('Submission exported.')




