import json
import os


class SubmissionWriter:

    def __init__(self):
        self.results = []
        return

    def append(self, filename, q, r):
        self.results.append({'filename': filename, 'q': q, 'r': r})
        return

    def export(self, out_dir='.'):
        sorted_results = sorted(self.results, key=lambda k: k['filename'])
        with open(os.path.join(out_dir, 'submission.json'), 'w') as f:
            json.dump(sorted_results, f)
        return

