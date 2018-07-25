import tensorflow as tf
import cv2
import numpy as np
import os
import time
import argparse
from glob import glob
import matplotlib.cm as cm
from utils.ops import non_max_suppression
from utils.parse_config import parse_config


class Inference(object):
    def __init__(self, cfg_file):
        cfg = parse_config(cfg_file)
        self.data_cfg = cfg['data_config']
        self.train_cfg = cfg['train_config']
        self.model_cfg = cfg['model_config']
        self.infer_cfg = cfg['infer_config']
        self.num_keypoints = len(self.train_cfg.train_keypoints)
        self.num_joints = len(self.train_cfg.train_skeletons)
        kp_dict = {kp: i for i, kp in
                   enumerate(self.train_cfg.train_keypoints)}
        self.pairs = [[kp_dict[kp1], kp_dict[kp2]] for kp1, kp2
                      in self.train_cfg.train_skeletons]
        self.col_channels = 3  # assume RGB channels only
        self.patch_h, self.patch_w = self.infer_cfg.network_input_shape
        self.out_stride = self.infer_cfg.out_stride
        self.frozen_model_file = os.path.join(
            self.infer_cfg.model_dir, self.infer_cfg.frozen_model)
        self.img_h, self.img_w = self.infer_cfg.resize_shape
        self.strides_rows, self.strides_cols = self.infer_cfg.strides
        self.n_rows = int(np.ceil(self.img_h / self.patch_h))
        self.n_cols = int(np.ceil(self.img_w / self.patch_w))
        with tf.device('/cpu:0'):
            self.preprocess_tensors = self.preprocess_graph()
        with tf.device('/gpu:0'):
            self.network_tensors = self.network_forward_pass()
        with tf.device('/cpu:0'):
            self.postprocess_tensors = self.postprocess_graph()

    def preprocess_image(self, image):
        # tensorflow expects RGB!
        image = image[:, :, ::-1]
        # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_w, self.img_h))
        if self.infer_cfg.flip_top_down:
            image = cv2.flip(image, 0)
        return image

    def create_patches(self, image):
        """tensorflow implementation for patch extraction"""
        images = tf.expand_dims(image, axis=0)
        patches = tf.extract_image_patches(
            images,
            ksizes=[1, self.patch_h, self.patch_w, 1],
            strides=[1, self.strides_rows, self.strides_cols, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
            name=None
        )
        patches = tf.reshape(
            patches,
            (self.n_rows * self.n_cols, self.patch_h,
             self.patch_w, self.col_channels))
        return patches

    def create_patches_v2(self, image):
        """custom implementation of patch extraction:
        [currently not used]
        NOTE: run speed similar to create_patches"""
        patches_top = np.repeat(
            self.strides_rows * np.arange(self.n_rows), self.n_cols)
        patches_left = np.tile(
            self.strides_cols * np.arange(self.n_cols), self.n_rows)

        def _extract_patch(patches_top_left_):
            y, x = patches_top_left_
            return image[y:y + self.patch_h, x:x + self.patch_w]

        patches = tf.map_fn(_extract_patch,
                            elems=(patches_top, patches_left),
                            back_prop=False,
                            parallel_iterations=12,  # CPU cores?
                            dtype=tf.float32)
        return patches

    def stitch_to_right(self, patch_left, patch_right, vector=False):
        """Creates heatmap by stitching patch_right on to patch_left
        Overlap region heatmap is the weighted sum of two patches"""
        overlap_cols = int(
            (self.patch_w - self.strides_cols) / self.out_stride)
        left, left_overlap = tf.split(
            patch_left, [-1, overlap_cols], axis=1)
        right_overlap, right = tf.split(
            patch_right, [overlap_cols, -1], axis=1)

        if vector:
            weights = tf.reshape(
                tf.range(overlap_cols, dtype=tf.float32),
                shape=(1, overlap_cols))
            overlap_vecmaps = []
            for i in range(2 * self.num_joints):
                left_vec_x = left_overlap[:, :, 2 * i]
                left_vec_y = left_overlap[:, :, 2 * i + 1]
                w_left = tf.clip_by_value(
                    overlap_cols - 1. - weights - left_vec_x,
                    clip_value_min=0,
                    clip_value_max=overlap_cols)
                w_left = tf.minimum(
                    w_left + 0.0001, overlap_cols - 1. - weights)
                right_vec_x = right_overlap[:, :, 2 * i]
                right_vec_y = right_overlap[:, :, 2 * i + 1]
                w_right = tf.clip_by_value(
                    weights + right_vec_x,
                    clip_value_min=0,
                    clip_value_max=overlap_cols)
                w_right = tf.minimum(w_right + 0.0001, weights)
                w_tot = w_left + w_right
                w_left = w_left / tf.maximum(w_tot, .0001)
                w_right = w_right / tf.maximum(w_tot, .0001)
                overlap_vec_x = (left_vec_x * w_left
                                 + right_vec_x * w_right)
                overlap_vec_y = (left_vec_y * w_left
                                 + right_vec_y * w_right)
                overlap_vecmaps += [overlap_vec_x, overlap_vec_y]
            overlap = tf.stack(overlap_vecmaps, axis=2)
        else:
            weights = tf.reshape(
                tf.range(overlap_cols, dtype=tf.float32),
                shape=(1, overlap_cols, 1))
            overlap = (left_overlap * (overlap_cols - 1. - weights)
                       + right_overlap * weights) / (overlap_cols - 1)
        out = tf.concat([left, overlap, right], axis=1)
        return out

    def stitch_to_bottom(self, patch_top, patch_bottom, vector=False):
        """Creates heatmap by stitching patch_bottom on to patch_top
        Overlap region heatmap is the weighted sum of two patches"""
        overlap_rows = int(
            (self.patch_h - self.strides_rows) / self.out_stride)
        top, top_overlap = tf.split(
            patch_top, [-1, overlap_rows], axis=0)
        bottom_overlap, bottom = tf.split(
            patch_bottom, [overlap_rows, -1], axis=0)

        if vector:
            weights = tf.reshape(
                tf.range(overlap_rows, dtype=tf.float32),
                shape=(overlap_rows, 1))
            overlap_vecmaps = []
            for i in range(2 * self.num_joints):
                top_vec_x = top_overlap[:, :, 2 * i]
                top_vec_y = top_overlap[:, :, 2 * i + 1]
                w_top = tf.clip_by_value(
                    overlap_rows - 1. - weights - top_vec_y,
                    clip_value_min=0,
                    clip_value_max=overlap_rows - 1)
                w_top = tf.minimum(
                    w_top + 0.0001, overlap_rows - 1. - weights)
                bottom_vec_x = bottom_overlap[:, :, 2 * i]
                bottom_vec_y = bottom_overlap[:, :, 2 * i + 1]
                w_bottom = tf.clip_by_value(
                    weights + bottom_vec_y,
                    clip_value_min=0,
                    clip_value_max=overlap_rows - 1)
                w_bottom = tf.minimum(w_bottom + 0.0001, weights)
                w_tot = w_top + w_bottom
                w_top = w_top / tf.maximum(w_tot, .0001)
                w_bottom = w_bottom / tf.maximum(w_tot, .0001)
                overlap_vec_x = (top_vec_x * w_top
                                 + bottom_vec_x * w_bottom)
                overlap_vec_y = (top_vec_y * w_top
                                 + bottom_vec_y * w_bottom)
                overlap_vecmaps += [overlap_vec_x, overlap_vec_y]
            overlap = tf.stack(overlap_vecmaps, axis=2)
        else:
            weights = tf.reshape(
                tf.range(overlap_rows, dtype=tf.float32),
                shape=(overlap_rows, 1, 1))
            overlap = (top_overlap * (overlap_rows - 1. - weights)
                       + bottom_overlap * weights) / (overlap_rows - 1)
        out = tf.concat([top, overlap, bottom], axis=0)
        return out

    def stitch_patches(self, patches, vector=False):
        rows = tf.split(
            patches, num_or_size_splits=self.n_rows, axis=0)
        stitched_rows = []
        for i, row in enumerate(rows):
            stitched_row = patches[i * self.n_cols]
            for j in range(1, self.n_cols):
                stitched_row = self.stitch_to_right(
                    stitched_row, patches[i * self.n_cols + j],
                    vector=vector)
            stitched_rows.append(stitched_row)

        out = stitched_rows[0]
        for i in range(1, self.n_rows):
            out = self.stitch_to_bottom(
                out, stitched_rows[i], vector=vector)

        out = tf.squeeze(out)
        return out

    def preprocess_graph(self):
        """Creates graph for preprocessing"""
        image = tf.placeholder(
            tf.float32,
            shape=[self.img_h, self.img_w, self.col_channels])
        patches = tf.expand_dims(image, axis=0)  # self.create_patches(image)
        return {'image': image,
                'patches': patches}

    def network_forward_pass(self):
        """Creates graph for network forward pass"""
        with tf.gfile.GFile(self.frozen_model_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.get_default_graph().as_default() as g:
            tf.import_graph_def(graph_def)
        patches = g.get_tensor_by_name('import/images:0')
        heatmaps = g.get_tensor_by_name('import/heatmaps:0')
        vecmaps = g.get_tensor_by_name('import/vecmaps:0')
        offsetmaps = g.get_tensor_by_name('import/offsetmaps:0')
        return {'patches': patches,
                'heatmaps': heatmaps,
                'vecmaps': vecmaps * self.train_cfg.vector_scale,
                'offsetmaps': offsetmaps * self.train_cfg.offset_scale}

    def postprocess_graph(self):
        """Creates graph for post processing"""
        heatmaps = tf.placeholder(
            tf.float32,
            shape=[self.n_rows * self.n_cols,
                   self.patch_h / self.out_stride,
                   self.patch_w / self.out_stride,
                   self.num_keypoints])
        vecmaps = tf.placeholder(
            tf.float32,
            shape=[self.n_rows * self.n_cols,
                   self.patch_h / self.out_stride,
                   self.patch_w / self.out_stride,
                   4 * self.num_joints])
        offsetmaps = tf.placeholder(
            tf.float32,
            shape=[self.n_rows * self.n_cols,
                   self.patch_h / self.out_stride,
                   self.patch_w / self.out_stride,
                   2 * self.num_keypoints])
        # stitched_heatmaps = self.stitch_patches(heatmaps)
        # heatmaps_nms = tf.expand_dims(stitched_heatmaps, axis=0)
        heatmaps_nms = heatmaps
        heatmaps_nms = non_max_suppression(
            heatmaps_nms, window_size=self.train_cfg.window_size)
        heatmaps_nms = tf.squeeze(heatmaps_nms)
        stitched_vecmaps = tf.squeeze(vecmaps)  # self.stitch_patches(vecmaps, vector=True)
        stitched_offsetmaps = tf.squeeze(offsetmaps)  # self.stitch_patches(offsetmaps)
        return {'heatmaps': heatmaps,
                'vecmaps': vecmaps,
                'offsetmaps': offsetmaps,
                'stitched_heatmaps': heatmaps_nms,
                'stitched_vecmaps': stitched_vecmaps,
                'stitched_offsetmaps': stitched_offsetmaps}

    def _run_inference(self, sess, image):
        t0 = time.time()
        patches = sess.run(
            self.preprocess_tensors['patches'],
            feed_dict={self.preprocess_tensors['image']: image})
        t1 = time.time()

        heatmaps, vecmaps, offsetmaps = sess.run(
            [self.network_tensors['heatmaps'],
             self.network_tensors['vecmaps'],
             self.network_tensors['offsetmaps']],
            feed_dict={self.network_tensors['patches']: patches})
        t2 = time.time()

        feed_dict = {
            self.postprocess_tensors['heatmaps']: heatmaps,
            self.postprocess_tensors['vecmaps']: vecmaps,
            self.postprocess_tensors['offsetmaps']: offsetmaps
        }
        stitched_maps = sess.run(
            [self.postprocess_tensors['stitched_heatmaps'],
             self.postprocess_tensors['stitched_vecmaps'],
             self.postprocess_tensors['stitched_offsetmaps']],
             feed_dict=feed_dict)
        t3 = time.time()
        return stitched_maps, [t0, t1, t2, t3]

    def display_output(self, image, heatmaps, vecmaps, offsetmaps,
                       pairs=([0, 1], [2, 1]), threshold=0.2):
        heatmaps[heatmaps > threshold] = 1.
        heatmaps[heatmaps <= threshold] = 0.
        img_h, img_w, _ = image.shape
        h, w, num_keypoints = heatmaps.shape
        scale_h, scale_w = img_h / h, img_w / w
        out_img = np.zeros((h, w, 3))
        colors = cm.jet(np.linspace(0, 1, num_keypoints))
        for i in range(num_keypoints):
            col = colors[i][:3]
            heatmap = heatmaps[:, :, i]
            heatmap = np.tile(np.expand_dims(heatmap, axis=2),
                              (1, 1, 3))
            out_img += heatmap * col.reshape((1, 1, 3))
        out_img = cv2.resize(out_img, (img_w, img_h),
                             interpolation=cv2.INTER_NEAREST)
        out_img = (255. * out_img).astype(np.uint8)
        # out_img = cv2.addWeighted(out_img, .5, image, 0.5, 0)
        out_img = cv2.addWeighted(out_img, 0.5, image[:, :, ::-1], 0.5, 0)
        for i, (kp1, kp2) in enumerate(pairs):
            y_indices_1, x_indices_1 = heatmaps[:, :, kp1].nonzero()
            for x, y in zip(x_indices_1, y_indices_1):
                x0 = int(scale_w * (x + .5))
                y0 = int(scale_h * (y + .5))
                delta_x = int(scale_w * (
                        vecmaps[y, x, 4 * i] + offsetmaps[y, x, kp2]))
                delta_y = int(scale_h * (
                        vecmaps[y, x, 4 * i + 1]
                        + offsetmaps[y, x, num_keypoints + kp2]))
                col = (255. * colors[kp1][:3]).astype(np.uint8)
                col = tuple(map(int, col))
                out_img = cv2.line(out_img, (x0, y0),
                                   (x0 + delta_x, y0 + delta_y),
                                   col, 1)
            y_indices_2, x_indices_2 = heatmaps[:, :, kp2].nonzero()
            for x, y in zip(x_indices_2, y_indices_2):
                x0 = int(scale_w * (x + .5))
                y0 = int(scale_h * (y + .5))
                delta_x = int(scale_w * (
                        vecmaps[y, x, 4 * i + 2] + offsetmaps[y, x, kp1]))
                delta_y = int(scale_h * (
                        vecmaps[y, x, 4 * i + 3]
                        + offsetmaps[y, x, num_keypoints + kp1]))
                col = (255. * colors[kp2][:3]).astype(np.uint8)
                col = tuple(map(int, col))
                out_img = cv2.line(out_img, (x0, y0),
                                   (x0 + delta_x, y0 + delta_y),
                                   col, 1)
        scale_ = 768. / min(img_h, img_w)
        out_img = cv2.resize(out_img, None, fx=scale_, fy=scale_)
        # out_img = out_img[:, :, ::-1]
        cv2.imshow('out', out_img)
        if cv2.waitKey(1) == 27:  # Esc key to stop
            return 0
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            return 0
        return 1

    def run_inference(self):
        sess = tf.Session()
        stats = SpeedStats()
        input_type = self.infer_cfg.input_type
        if input_type == 'images':
            img_files = self.infer_cfg.images
            if not isinstance(img_files, list):
                img_files = glob(img_files)
                img_files.sort()
            for img_file in img_files:
                image = cv2.imread(img_file)
                image = self.preprocess_image(image)
                stitched_maps, t = self._run_inference(sess, image)
                stats.update(t)
                heatmaps, vecmaps, offsetmaps = stitched_maps
                out = self.display_output(
                    image, heatmaps, vecmaps, offsetmaps,
                    pairs=self.pairs, threshold=0.3)
                if not out:
                    break
        elif input_type == 'video':
            video_file = self.infer_cfg.video
            cap = cv2.VideoCapture(video_file)
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break
                image = self.preprocess_image(image)
                stitched_maps, t = self._run_inference(sess, image)
                stats.update(t)
                heatmaps, vecmaps, offsetmaps = stitched_maps
                out = self.display_output(
                    image, heatmaps, vecmaps, offsetmaps,
                    pairs=self.pairs, threshold=0.15)
                if not out:
                    break
            cap.release()
        elif input_type == 'camera':
            cam_url = self.infer_cfg.camera
            cap = cv2.VideoCapture(cam_url)
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break
                image = self.preprocess_image(image)
                stitched_maps, t = self._run_inference(sess, image)
                stats.update(t)
                heatmaps, vecmaps, offsetmaps = stitched_maps
                out = self.display_output(
                    image, heatmaps, vecmaps, offsetmaps,
                    pairs=self.pairs, threshold=0.15)
                if not out:
                    break
            cap.release()
        else:
            raise RuntimeError(
                "input type {} not supported".format(input_type))

        cv2.destroyAllWindows()
        stats.summarize()


class SpeedStats(object):
    def __init__(self):
        self.sum_t = [0., 0., 0.]
        self.n_skip = 3
        self.count = 0

    def update(self, t):
        if self.count > self.n_skip:
            self.sum_t[0] += t[1] - t[0]
            self.sum_t[1] += t[2] - t[1]
            self.sum_t[2] += t[3] - t[2]
        self.count += 1

    def summarize(self):
        print("Pre-processing time : {:5.2f} ms/image".format(
            self.sum_t[0] / (self.count - self.n_skip) * 1000))
        print("Network inference time : {:5.2f} ms/image".format(
            self.sum_t[1] / (self.count - self.n_skip) * 1000))
        print("Post-processing time : {:5.2f} ms/image".format(
            self.sum_t[2] / (self.count - self.n_skip) * 1000))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str,
                        default='./config.yaml', help='Config file')
    args = parser.parse_args()
    config_file = args.config_file
    assert os.path.exists(config_file), \
        "{} not found".format(config_file)
    infer = Inference(config_file)
    infer.run_inference()
