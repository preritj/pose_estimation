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
    def __init__(self, image_dir, annotation_file=None):
        """
        Constructor of Pose class for reading and visualizing annotations
         from human pose datasets.
        :param annotation_file (str): location of annotation file
        :param image_dir (str): location of image directory
        :return:
        """
        self.image_dir = image_dir
        self.num_keypoints = 15
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
        if annotation_file is not None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
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
    def __init__(self, image_dir, annotation_file=None):
        super().__init__(image_dir, annotation_file)

    def create_index(self):
        # create index
        print('creating index...')
        imgs, anns, masks = {}, defaultdict(list), defaultdict(list)

        for img in self.dataset['images']:
            imgs[img['id']] = {'file_name': img['file_name'],
                               'height': img['height'],
                               'width': img['width']}

        kp_names = self.dataset['categories'][0]['keypoints']

        for ann in tqdm(self.dataset['annotations']):
            if ann['category_id'] != 1:
                continue
            area = ann['area']
            img = imgs[ann['image_id']]
            img_area = img['height'] * img['width']
            num_keypoints = ann['num_keypoints']
            if ann['iscrowd'] or (area > .25 * img_area) or (num_keypoints < 2):
                ignore_region = ann['segmentation']
                masks[ann['image_id']].append(
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
                keypoints[2, 2] = np.int(keypoints[2, 2])
            anns[ann['image_id']].append({'keypoints': keypoints,
                                          'bbox': bbox})

        print('index created!')

        # create class members
        self.anns = anns
        self.imgs = imgs
        self.ids = list(anns.keys())
        self.masks = masks


class MPII(Pose):
    def __init__(self, image_dir, annotation_file=None):
        super().__init__(image_dir, annotation_file)

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
        self.dataset = self.dataset['RELEASE'][0]
        is_train_list = self.dataset['img_train']

        imgs, anns = {}, defaultdict(list)
        for i, (is_train, annotations) in enumerate(
                zip(is_train_list, self.dataset['annolist'])):
            imgs[i] = annotations['image'][0]['name']
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
                    v = v[0] if len(v) > 0 else -1
                    kp_name = kp_dict[id_]
                    if kp_name in self.keypoints.keys():
                        kp_idx = self.keypoints[kp_name]
                        keypoints_clean[kp_idx] = [x, y, v]
                anns[i].append({'keypoints': keypoints_clean})

        print('index created!')

        # create class members
        self.anns = anns
        self.imgs = imgs
        self.ids = anns.keys()
