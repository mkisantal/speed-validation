import numpy as np
import xml.etree.ElementTree as ET
import os
from PIL import Image


def xmlread(path):

    """ Getting root of xml tree. """

    tree = ET.parse(path)
    root = tree.getroot()
    return root


def parse_path(image):

    """ Parsing path from xml image object. """

    file_name = image.findall('imageFileName')[0].text
    path = os.path.join('/datasets/speed_v1/data/speed/images', file_name)
    return path


def parse_file_name(image):

    """ Parsing file name from xml image object. """

    file_name = image.findall('imageFileName')[0].text
    return file_name


def parse_pose(image):

    """ Parsing pose variables, both ground truth and prediction. """

    q_vbs2tango_true = np.array([float(i) for i in image.findall('q_vbs2tango_true')[0].text.split(',')])
    r_Vo2To_vbs_true = np.array([float(i) for i in image.findall('r_Vo2To_vbs_true')[0].text.split(',')])

    # we might not have estimated values yet
    try:
        q_vbs2tango_est = np.array([float(i) for i in image.findall('q_vbs2tango_est')[0].text.split(',')])
        r_Vo2To_vbs_est = np.array([float(i) for i in image.findall('r_Vo2To_vbs_est')[0].text.split(',')])
    except IndexError:
        return q_vbs2tango_true, r_Vo2To_vbs_true

    return q_vbs2tango_true, r_Vo2To_vbs_true, q_vbs2tango_est, r_Vo2To_vbs_est


def process_xml_dataset(root_dir='/datasets/speed_v1/data/speed/'):

    """ Parsing xml dataset, both train and test sets. """

    labels = {}
    partition = {}

    subsets = ['train', 'test']
    for subset in subsets:
        partition[subset] = []

        xml_set = xmlread(os.path.join(root_dir, '{}.xml'.format(subset)))

        for image in xml_set.findall('image'):
            q_vbs2tango_true, r_Vo2To_vbs_true = parse_pose(image)
            id = parse_file_name(image)[:-4]
            partition[subset].append(id)
            labels[id] = {'q': q_vbs2tango_true, 'r': r_Vo2To_vbs_true}

    return partition, labels


def process_xml_tron_dataset(tron_root='/datasets/tronRealImages'):

    """ Parsing xml dataset for tron images. """

    labels = {}
    partition = {}

    subset = 'tron'
    partition[subset] = []

    xml_set = xmlread(os.path.join(tron_root, 'tronRealImages.xml'))
    for image in xml_set.findall('image'):
        q_vbs2tango_true, r_Vo2To_vbs_true = parse_pose(image)
        id = parse_file_name(image)[:-4] + 'tron'
        partition[subset].append(id)
        labels[id] = {'q': q_vbs2tango_true, 'r': r_Vo2To_vbs_true}

    return partition, labels


class SatellitePoseEstimationDataset:

    """
        Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data.
        Parses original speed_v1 .xml files, with pose label for both train and test sets.
    """

    def __init__(self, root_dir='/datasets/speed_v1/data/speed/', tron=False):
        if tron:
            self.partitions, self.labels = process_xml_tron_dataset(root_dir)
        else:
            self.partitions, self.labels = process_xml_dataset(root_dir)
        self.root_dir = root_dir

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_id = self.partitions[split][i]
        img_name = os.path.join(self.root_dir, 'images', img_id + '.jpg')
        image = Image.open(img_name).convert('RGB')
        return image

    def get_pose(self, i=0, split='train'):

        """ Getting pose label for image. """

        img_id = self.partitions[split][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r
