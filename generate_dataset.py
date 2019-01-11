import os
import json
from shutil import copyfile
import speed_utils

"""
    Generating restructured dataset from speed_v1:
        - splitting images to separate folders
        - generating json files instead the source of xml
        - saving test images without labels
        - saving test labels separately
"""

speed = speed_utils.SatellitePoseEstimationDataset()
tron = speed_utils.SatellitePoseEstimationDataset(root_dir='/datasets/tronRealImages', tron=True)


source_root = speed.root_dir
destination_root = '/datasets/speed_debug2'
test_labels_root = '/datasets/speed_debug2_TEST_LABELS'


dst_images = os.path.join(destination_root, 'images')
if not os.path.exists(dst_images):
    os.makedirs(dst_images)
if not os.path.exists(test_labels_root):
    os.makedirs(test_labels_root)
for partition in ['train', 'test', 'tron']:
    if not os.path.exists(os.path.join(dst_images, partition)):
        os.makedirs(os.path.join(dst_images, partition))


# split images to folders
for partition in ['train', 'test']:
    # sort the images
    speed.partitions[partition].sort()
    for image_id in speed.partitions[partition][:100]:
        src = os.path.join(source_root, 'images', image_id + '.jpg')
        dst = os.path.join(destination_root, 'images', partition, image_id + '.jpg')
        copyfile(src, dst)

for image_id in tron.partitions['tron']:
    src = os.path.join(tron.root_dir, 'images', image_id[:-4] + '.jpg')
    dst = os.path.join(destination_root, 'images', 'tron', image_id  + '.jpg')
    copyfile(src, dst)

# saving json
for partition in ['train', 'test']:
    images = []
    images_with_labels = []
    for image_id in speed.partitions[partition][:20]:
        image_dict = dict()
        image_with_label = dict()
        image_dict['filename'] = '{}.jpg'.format(image_id)
        image_with_label['filename'] = '{}.jpg'.format(image_id)
        images.append(image_dict)

        # filename was stored above, now we continue adding pose label too
        image_with_label['q_vbs2tango'] = list(speed.labels[image_id]['q'])
        image_with_label['r_Vo2To_vbs_true'] = list(speed.labels[image_id]['r'])
        images_with_labels.append(image_with_label)
    json_path = os.path.join(destination_root, '{}.json'.format(partition))
    if partition == 'train':
        with open(json_path, 'w') as f:
            json.dump(images_with_labels, f)
    else:
        with open(json_path, 'w') as f:
            json.dump(images, f)
        with open(os.path.join(test_labels_root, 'test_labels.json'), 'w') as f:
            json.dump(images_with_labels, f)

# saving json for tron
images = []
images_with_labels = []
for image_id in tron.partitions['tron'][:20]:
    image_dict = dict()
    image_with_label = dict()
    image_dict['filename'] = '{}.jpg'.format(image_id)
    image_with_label['filename'] = '{}.jpg'.format(image_id)
    images.append(image_dict)

    image_with_label['q_vbs2tango'] = list(tron.labels[image_id]['q'])
    image_with_label['r_Vo2To_vbs_true'] = list(tron.labels[image_id]['r'])
    images_with_labels.append(image_with_label)

with open(os.path.join(destination_root, '{}.json'.format('tron')), 'w') as f:
    json.dump(images, f)
with open(os.path.join(test_labels_root, 'tron_labels.json'), 'w') as f:
    json.dump(images_with_labels, f)

print('Dataset created at {}'.format(destination_root))
print('Ground truth for test set was saved separately at {}.'.format(test_labels_root))
