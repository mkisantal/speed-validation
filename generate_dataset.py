import sys
import os
import json
from shutil import copyfile
import speed_utils

"""
    Generating restructured dataset from speed_v1:
        - splitting images to separate folders
        - generating json files instead the source of xml
        - saving test images without labels
"""

speed = speed_utils.SatellitePoseEstimationDataset()

source_root = speed.root_dir
destination_root = '/datasets/speed_debug'

dst_images = os.path.join(destination_root, 'images')
if not os.path.exists(dst_images):
    os.makedirs(dst_images)
for partition in ['train', 'test']:
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

# saving json
for partition in ['train', 'test']:
    label_list = []
    for image_id in speed.partitions[partition][:20]:
        image_label = {}
        image_label['filename'] = '{}.jpg'.format(image_id)
        if partition == 'train':
            image_label['q_vbs2tango'] = list(speed.labels[image_id]['q'])
            image_label['r_Vo2To_vbs_true'] = list(speed.labels[image_id]['r'])
        label_list.append(image_label)
    json_path = os.path.join(destination_root, '{}.json'.format(partition))
    with open(json_path, 'w') as f:
        json.dump(label_list, f)

print('Dataset created at {}'.format(destination_root))

