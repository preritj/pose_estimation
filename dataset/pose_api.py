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

    def display_anns(self, img_id):
        filename = self.imgs[img_id]['file_name']
        img_file = os.path.join(self.image_dir, filename)
        img = plt.imread(img_file)
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
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)
        return m


class COCO(Pose):
    def __init__(self, image_dir, annotation_files=None):
        super().__init__(image_dir, annotation_files)

    def _build_dataset(self, dataset):
        self.dataset_count += 1
        for img in dataset['images']:
            self.imgs[img['id']] = {'file_name': img['file_name'],
                                    'height': img['height'],
                                    'width': img['width']}

        kp_names = dataset['categories'][0]['keypoints']

        for ann in tqdm(dataset['annotations']):
            if ann['category_id'] != 1:
                continue
            area = ann['area']
            img = self.imgs[ann['image_id']]
            img_area = img['height'] * img['width']
            num_keypoints = ann['num_keypoints']
            if ann['iscrowd'] or (area > .25 * img_area) or (num_keypoints < 2):
                ignore_region = ann['segmentation']
                self.masks[ann['image_id']].append(
                    {'ignore_region': ignore_region})
                continue
            bbox = ann['bbox']
            kp = np.array(ann['keypoints'])
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]
            keypoints = np.zeros((self.num_keypoints, 3))

            for kp_name, x, y, v in zip(kp_names, xs, ys, vs):
                if kp_name not in self.keypoints.keys():
                    continue
                kp_idx = self.keypoints[kp_name]
                keypoints[kp_idx] = [x, y, v]
            # hack for neck in COCO
            if (vs[5] > 0) and (vs[6] > 0):
                keypoints[2] = np.mean(keypoints[3:5], axis=0)
                if vs[0] > 0:
                    keypoints[2] = (keypoints[1] + 3. * keypoints[2]) / 4.
                elif vs[3] > 0 and vs[4] > 0:
                    face_center = np.array([np.mean(xs[3:5]), np.mean(ys[3:5]), 1])
                    keypoints[2] = (face_center + 3. * keypoints[2]) / 4.
                elif vs[3] > 0:
                    face_center = np.array([xs[3], ys[3], 1])
                    keypoints[2] = (face_center + 3. * keypoints[2]) / 4.
                elif vs[4] > 0:
                    face_center = np.array([xs[4], ys[4], 1])
                    keypoints[2] = (face_center + 3. * keypoints[2]) / 4.
                keypoints[2, 2] = np.int(keypoints[2, 2])
            # hack for head top
            face_center = [-1, -1]
            bottom = keypoints[2]
            if vs[3] > 0 and vs[4] > 0:
                face_center = [np.mean(xs[3:5]), np.mean(ys[3:5])]
            elif vs[1] > 0 and vs[2] > 0:
                face_center = [np.mean(xs[1:3]), np.mean(ys[1:3])]
                if vs[0] > 0 and bottom[2] > 0:
                    bottom = (4. * np.array([xs[0], ys[0], vs[0]]) + bottom) / 5.
            elif vs[1] > 0 and vs[3] > 0:
                face_center = [np.mean(xs[[1, 3]]), np.mean(ys[[1, 3]])]
                if vs[0] > 0 and bottom[2] > 0:
                    bottom = (np.array([xs[0], ys[0], vs[0]]) + 2. * bottom) / 3.
            elif vs[2] > 0 and vs[4] > 0:
                face_center = [np.mean(xs[[2, 4]]), np.mean(ys[[2, 4]])]
                if vs[0] > 0 and bottom[2] > 0:
                    bottom = (np.array([xs[0], ys[0], vs[0]]) + 2. * bottom) / 3.
            if bottom[2] > 0 and np.all(np.array(face_center) > 0):
                # estimate using neck
                head_x = 2. * face_center[0] - bottom[0]
                head_y = 2. * face_center[1] - bottom[1]
                keypoints[0] = [head_x, head_y, 1]
            self.anns[ann['image_id']].append({'keypoints': keypoints,
                                               'bbox': bbox})

    def create_index(self):
        # create index
        print('creating index...')
        self.imgs, self.anns, self.masks = {}, defaultdict(list), defaultdict(list)

        for dataset in self.datasets:
            self._build_dataset(dataset)

        print('index created!')
        self.ids = list(self.anns.keys())


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
            img_id = self.dataset_count*10000 + annotations['imgnum'][0]
            self.imgs[img_id] = {'file_name': img_name,
                                 'height': None,
                                 'width': None}
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

        for dataset in self.datasets:
            self._build_dataset(dataset, kp_dict)

        print('index created!')
        self.ids = list(self.anns.keys())

