

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
    if file.size > 15 * (1 << 20):
        raise ValueError('File size too big, maximum is 15 MB.')
