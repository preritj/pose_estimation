import time
import json
import numpy as np
import os
import io
from abc import abstractmethod
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pycocotools import mask as maskUtils
import cv2
import PIL.Image
from tqdm import tqdm
from utils import dataset_util
import tensorflow as tf


class PoseData(object):
    def __init__(self, image_dir, annotation_files=None):
        """
        Constructor of Pose class for reading and visualizing annotations
         from human pose datasets.
        :param annotation_files (str or list): list of annotation files
        :param image_dir (str): location of image directory
        :param save_ann_dir (str): location where indexed annotations are saved
        :return:
        """
        self.image_dir = image_dir
        assert os.path.exists(image_dir), "Image directory not found"
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
        self.sigma = 10.
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

    def _create_tf_example(self, img_id):
        img_meta = self.imgs[img_id]
        img_file = img_meta['filename']
        img_file = os.path.join(self.image_dir, img_file)
        img_shape = list(img_meta['shape'])
        bboxes_bytes = []
        keypoints_bytes = []
        for ann in self.anns[img_id]:
            bboxes_bytes.append(np.array(ann['bbox']).tostring())
            keypoints_bytes.append(ann['keypoints'])
        mask = self.get_mask(img_id)
        if mask is None:
            mask = np.zeros(img_shape, dtype=np.uint8)
        n_instances = len(keypoints_bytes)
        keypoints_bytes = np.array(keypoints_bytes).flatten().tolist()

        img = PIL.Image.fromarray(mask)
        output_io = io.BytesIO()
        img.save(output_io, format='PNG')
        mask_bytes = output_io.getvalue()

        feature_dict = {
            'image/filename':
                dataset_util.bytes_feature(img_file.encode('utf8')),
            'image/shape':
                dataset_util.int64_list_feature(img_shape),
            'image/num_instances':
                dataset_util.int64_feature(n_instances),
            'image/person/bbox':
                dataset_util.bytes_list_feature(bboxes_bytes),
            'image/person/keypoints':
                dataset_util.float_list_feature(keypoints_bytes),
            'image/mask':
                dataset_util.bytes_feature(mask_bytes)}
        return tf.train.Example(features=tf.train.Features(feature=feature_dict))

    def create_tf_record(self, out_path, shuffle=True):
        print("Creating tf records : ", out_path)
        writer = tf.python_io.TFRecordWriter(out_path)
        if shuffle:
            np.random.shuffle(self.ids)
        for img_id in tqdm(self.ids):
            tf_example = self._create_tf_example(img_id)
            writer.write(tf_example.SerializeToString())
        writer.close()

    def load_annotations(self, ann_dir, ext='.jpg'):
        meta = []
        for root, dirs, files in os.walk(ann_dir):
            for file in files:
                if file.endswith(".npy"):
                    ann_file = os.path.join(root, file)
                    rel_path = os.path.relpath(root, ann_dir)
                    img_file = file.split('.')[0] + ext
                    img_file = os.path.join(self.image_dir, rel_path, img_file)
                    meta.append((img_file, ann_file))
        return meta

    @staticmethod
    def _generate_heatmap(center, sigma, shape):
        heatmap = np.zeros(shape)
        roi_min = np.maximum(np.array(center) - 2 * sigma, 0).astype(np.int)
        roi_max = np.minimum(np.array(center) + 2 * sigma, list(shape)).astype(np.int)
        x = np.arange(roi_min[0], roi_max[0])
        y = np.arange(roi_min[1], roi_max[1])
        x, y = np.meshgrid(x, y)
        d = (x - center[0]) ** 2 + (y - center[1]) ** 2
        heatmap[y, x] = np.exp(- d / sigma / sigma)
        return heatmap

    @staticmethod
    def get_mask_rles(img_shape, masks):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: a list of rles
        """
        h, w = img_shape
        seg_masks = masks
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

    def get_heatmap(self, img_id, out_shape=None):
        h, w = self.imgs[img_id]['shape']
        heatmaps = np.zeros((h, w, self.num_keypoints))
        for ann in self.anns[img_id]:
            keypoints = ann['keypoints']
            for i, kp in enumerate(keypoints):
                if kp[2] < 1:
                    continue
                heatmap = self._generate_heatmap(kp[:2], self.sigma, [h, w])
                heatmaps[:, :, i] = np.maximum(heatmaps[:, :, i], heatmap)
        if out_shape is not None:
            map_h, map_w = out_shape
            heatmaps = cv2.resize(heatmaps, (map_w, map_h),
                                  interpolation=cv2.INTER_AREA)
        return heatmaps

    def get_mask(self, img_id):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        img_shape = self.imgs[img_id]['shape']
        rles = self.get_mask_rles(img_shape, self.masks[img_id])
        if len(rles) == 0:
            return None
        rle = maskUtils.merge(rles)
        m = maskUtils.decode(rle)
        return m

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

