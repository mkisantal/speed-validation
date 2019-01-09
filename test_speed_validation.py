import speed_validation

# Todo: use django objects for passing the file
# from django.db import models
# from django.db.models.fields.files import FieldFile
# file = FieldFile(None, None, None)
# print(file)
# file.open('submission_debug.json')
# print(file.size)


submission_file = open('submission_debug.json', 'r')
# submission_file = open('not_a_real.json', 'r')
res = speed_validation.validate(submission_file)
print('validate:\n{}'.format(res))
res = speed_validation.score(submission_file)
print('score:\n{}'.format(res))

submission_file.close()
