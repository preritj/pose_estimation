import time
import json
import numpy as np
import os
from tqdm import tqdm
from abc import abstractmethod
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pycocotools import mask as maskUtils
from PIL import Image
import cv2


class Pose:
    def __init__(self, image_dir, annotation_files=None):
        """
        Constructor of Pose class for reading and visualizing annotations
         from human pose datasets.
        :param annotation_files (str): list of annotation files
        :param image_dir (str): location of image directory
        :return:
        """
        self.image_dir = image_dir
        self.num_keypoints = 15
        self.dataset_count = -1
        self.keypoints = {'head': 0,
                          'nose': 1,
                          'neck': 2,
                          'left_shoulder': 3,
                          'right_shoulder': 4,
                          'left_elbow': 5,
                          'right_elbow': 6,
                          'left_wrist': 7,
                          'right_wrist': 8,
                          'left_hip': 9,
                          'right_hip': 10,
                          'left_knee': 11,
                          'right_knee': 12,
                          'left_ankle': 13,
                          'right_ankle': 14}
        self.skeleton = [[0, 1], [1, 2], [2, 3], [2, 4], [2, 9], [2, 10], [3, 5],
                         [4, 6], [5, 7], [6, 8], [9, 11], [10, 12], [11, 13], [12, 14]]
        self.imgs, self.ids, self.anns, self.masks = None, None, None, None
        if annotation_files is not None:
            print('loading annotations into memory...')
            tic = time.time()
            self.datasets = []
            if type(annotation_files) != list:
                annotation_files = [annotation_files]
            for ann_file in annotation_files:
                dataset = json.load(open(ann_file, 'r'))
                assert type(dataset) == dict, \
                    'annotation file format {} not supported'.format(type(dataset))
                self.datasets.append(dataset)
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.create_index()

    @abstractmethod
    def create_index(self):
        return

    def display_anns(self, img_id, show_mask=False):
        filename = self.imgs[img_id]['file_name']
        img_file = os.path.join(self.image_dir, filename)
        img = plt.imread(img_file)
        if show_mask:
            mask_img = np.zeros_like(img)
            mask = self.get_mask(img_id)
            if mask is not None:
                mask_img[:, :, 0] = 255. * mask
                img = cv2.addWeighted(img, 1., mask_img, 0.5, 0)
        plt.imshow(img)
        sks = np.array(self.skeleton)
        colors = cm.jet(np.linspace(0, 1, self.num_keypoints))
        np.random.seed(1999)
        np.random.shuffle(colors)
        for ann in self.anns[img_id]:
            kp = ann['keypoints']
            x, y, v = kp[:, 0], kp[:, 1], kp[:, 2]
            for c, sk in zip(colors, sks):
                if np.all(v[sk] > 0):
                    plt.plot(x[sk], y[sk], linewidth=3, color=c)

    def get_mask_rles(self, img_id):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: a list of rles
        """
        t = self.imgs[img_id]
        h, w = t['height'], t['width']
        seg_masks = self.masks[img_id]
        rle_list = []
        for seg_mask in seg_masks:
            segm = seg_mask['ignore_region']
            if type(segm) == list:
                # polygon -- a single object might consist of multiple parts
                # we merge all parts into one mask rle code
                rles = maskUtils.frPyObjects(segm, h, w)
                rle = maskUtils.merge(rles)
            elif type(segm['counts']) == list:
                # uncompressed RLE
                rle = maskUtils.frPyObjects(segm, h, w)
            else:
                # rle
                rle = segm
            rle_list.append(rle)
        return rle_list

    def get_mask(self, img_id):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rles = self.get_mask_rles(img_id)
        if len(rles) == 0:
            return None
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)
        return m





class MPII(Pose):
    def __init__(self, image_dir, annotation_files=None):
        super().__init__(image_dir, annotation_files)

    def _build_dataset(self, dataset, kp_dict):
        self.dataset_count += 1
        dataset = dataset['RELEASE'][0]
        is_train_list = dataset['img_train']

        for i, (is_train, annotations) in enumerate(
                zip(is_train_list, dataset['annolist'])):
            img_name = annotations['image'][0]['name']
            img_id = self.dataset_count*50000 + i
            self.imgs[i] = {'file_name': img_name,
                            'height': None,
                            'width': None}
            if not is_train:
                continue
            persons = annotations['annorect']
            for person in persons:
                if 'annopoints' not in person.keys():
                    continue
                if len(person['annopoints']) == 0:
                    continue
                keypoints = person['annopoints'][0]['point']
                keypoints_clean = np.zeros((self.num_keypoints, 3))
                for kp in keypoints:
                    if 'is_visible' not in kp.keys():
                        kp['is_visible'] = [-1]
                    id_, x, y, v = (kp['id'][0], kp['x'][0],
                                    kp['y'][0], kp['is_visible'])
                    v = v[0] if len(v) > 0 else 2
                    kp_name = kp_dict[id_]
                    if kp_name in self.keypoints.keys():
                        kp_idx = self.keypoints[kp_name]
                        keypoints_clean[kp_idx] = [x, y, v]
                # hack for nose in MPII
                if keypoints_clean[0, 2] > 0 and keypoints_clean[2, 2] > 0:
                    keypoints_clean[1] = np.mean(keypoints_clean[[0, 2], :], axis=0)
                self.anns[i].append({'keypoints': keypoints_clean})

    def create_index(self):
        kp_dict = {0: 'right_ankle',
                   1: 'right_knee',
                   2: 'right_hip',
                   3: 'left_hip',
                   4: 'left_knee',
                   5: 'left_ankle',
                   6: 'pelvis',
                   7: 'thorax',
                   8: 'neck',
                   9: 'head',
                   10: 'right_wrist',
                   11: 'right_elbow',
                   12: 'right_shoulder',
                   13: 'left_shoulder',
                   14: 'left_elbow',
                   15: 'left_wrist'}
        # create index
        print('creating index...')
        self.imgs, self.anns, self.masks = {}, defaultdict(list), defaultdict(list)

        for dataset in self.datasets:
            self._build_dataset(dataset, kp_dict)

        print('index created!')
        self.ids = list(self.anns.keys())


class PoseTrack(Pose):
    def __init__(self, image_dir, annotation_files=None):
        super().__init__(image_dir, annotation_files)

    def _build_dataset(self, dataset, kp_dict):
        self.dataset_count += 1
        for annotations in dataset['annolist']:
            img_name = annotations['image'][0]['name']
            img_path = os.path.join(self.image_dir, img_name)
            im = Image.open(img_path)
            width, height = im.size
            img_id = self.dataset_count*10000 + annotations['imgnum'][0]
            self.imgs[img_id] = {'file_name': img_name,
                                 'height': height,
                                 'width': width}
            if not annotations['is_labeled']:
                continue
            persons = annotations['annorect']
            for person in persons:
                if 'annopoints' not in person.keys():
                    continue
                if len(person['annopoints']) == 0:
                    continue
                keypoints = person['annopoints'][0]['point']
                keypoints_clean = np.zeros((self.num_keypoints, 3))
                for kp in keypoints:
                    if 'is_visible' not in kp.keys():
                        kp['is_visible'] = [-1]
                    id_, x, y, v = (kp['id'][0], kp['x'][0],
                                    kp['y'][0], kp['is_visible'])
                    v = v[0] if len(v) > 0 else 2
                    kp_name = kp_dict[id_]
                    if kp_name in self.keypoints.keys():
                        kp_idx = self.keypoints[kp_name]
                        keypoints_clean[kp_idx] = [x, y, v]
                self.anns[img_id].append({'keypoints': keypoints_clean})

            if 'ignore_regions' in annotations.keys():
                ignore_regions = annotations['ignore_regions']
                polygons = []
                for reg in ignore_regions:
                    polygon = []
                    polygon_pts = reg['point']
                    for point in polygon_pts:
                        polygon.append(point['x'][0])
                        polygon.append(point['y'][0])
                    polygons.append(polygon)
                self.masks[img_id].append({'ignore_region': polygons})

    def create_index(self):
        kp_dict = {0: 'right_ankle',
                   1: 'right_knee',
                   2: 'right_hip',
                   3: 'left_hip',
                   4: 'left_knee',
                   5: 'left_ankle',
                   6: 'right_wrist',
                   7: 'right_elbow',
                   8: 'right_shoulder',
                   9: 'left_shoulder',
                   10: 'left_elbow',
                   11: 'left_wrist',
                   12: 'neck',
                   13: 'nose',
                   14: 'head'}
        # create index
        print('creating index...')
        self.imgs, self.anns, self.masks = {}, defaultdict(list), defaultdict(list)

        for dataset in tqdm(self.datasets):
            self._build_dataset(dataset, kp_dict)

        print('index created!')
        self.ids = list(self.anns.keys())

