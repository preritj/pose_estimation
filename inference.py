import tensorflow as tf
import cv2
import numpy as np
import os
import time
import argparse
from glob import glob
from utils.ops import non_max_suppression
from utils.bboxes import generate_anchors
from utils.parse_config import parse_config


class Inference(object):
    def __init__(self, cfg_file):
        cfg = parse_config(cfg_file)
        self.data_cfg = cfg['data_config']
        self.train_cfg = cfg['train_config']
        self.model_cfg = cfg['model_config']
        self.infer_cfg = cfg['infer_config']
        self.col_channels = 3  # assume RGB channels only
        self.patch_h, self.patch_w = self.infer_cfg.network_input_shape
        self.out_stride = self.infer_cfg.out_stride
        self.frozen_model_file = os.path.join(
            self.infer_cfg.model_dir, self.infer_cfg.frozen_model)
        self.img_h, self.img_w = self.infer_cfg.resize_shape
        self.strides_rows, self.strides_cols = self.infer_cfg.strides
        self.n_rows = int(np.ceil(self.img_h / self.patch_h))
        self.n_cols = int(np.ceil(self.img_w / self.patch_w))
        self.preprocess_tensors = self.preprocess_graph()
        self.network_tensors = self.network_forward_pass()
        self.postprocess_tensors = self.postprocess_graph()

    def preprocess_image(self, image):
        # tensorflow expects RGB!
        image = image[:, :, ::-1]
        # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_w, self.img_h))
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

    def stitch_to_right(self, patch_left, patch_right):
        """Creates heatmap by stitching patch_right on to patch_left
        Overlap region heatmap is the weighted sum of two patches"""
        overlap_cols = int(
            (self.patch_w - self.strides_cols) / self.out_stride)
        left, left_overlap = tf.split(
            patch_left, [-1, overlap_cols], axis=1)
        right_overlap, right = tf.split(
            patch_right, [overlap_cols, -1], axis=1)
        weights = tf.reshape(
            tf.range(overlap_cols, dtype=tf.float32),
            shape=(1, overlap_cols, 1))
        overlap = (left_overlap * (overlap_cols - 1. - weights)
                   + right_overlap * weights) / overlap_cols
        out = tf.concat([left, overlap, right], axis=1)
        return out

    def stitch_to_bottom(self, patch_top, patch_bottom):
        """Creates heatmap by stitching patch_bottom on to patch_top
        Overlap region heatmap is the weighted sum of two patches"""
        overlap_rows = int(
            (self.patch_h - self.strides_rows) / self.out_stride)
        top, top_overlap = tf.split(
            patch_top, [-1, overlap_rows], axis=0)
        bottom_overlap, bottom = tf.split(
            patch_bottom, [overlap_rows, -1], axis=0)
        weights = tf.reshape(
            tf.range(overlap_rows, dtype=tf.float32),
            shape=(overlap_rows, 1, 1))
        overlap = (top_overlap * (overlap_rows - 1. - weights)
                   + bottom_overlap * weights) / overlap_rows
        out = tf.concat([top, overlap, bottom], axis=0)
        return out

    def stitch_patches(self, patches):
        rows = tf.split(
            patches, num_or_size_splits=self.n_rows, axis=0)
        stitched_rows = []
        for i, row in enumerate(rows):
            stitched_row = patches[i * self.n_cols]
            for j in range(1, self.n_cols):
                stitched_row = self.stitch_to_right(
                    stitched_row, patches[i * self.n_cols + j])
            stitched_rows.append(stitched_row)

        out = stitched_rows[0]
        for i in range(1, self.n_rows):
            out = self.stitch_to_bottom(out, stitched_rows[i])
        out = tf.expand_dims(out, axis=0)
        out = non_max_suppression(out, window_size=3)
        out = tf.squeeze(out)
        return out

    # def get_bboxes(self, patches, bbox_probs, bbox_regs):
    #     for i in range(self.n_rows * self.n_cols):
    #         indices = tf.squeeze(tf.where(
    #             tf.greater(bbox_probs[i], 0.5)))
    #
    #         def _draw_bboxes():
    #             img = tf.squeeze(patches[i])
    #             bboxes = tf.gather(bbox_regs[i], indices)
    #             # bboxes = tf.zeros_like(bboxes)
    #             anchors = tf.gather(all_anchors, indices)
    #             bboxes = bbox_decode(
    #                 bboxes, anchors, self.model_cfg.scale_factors)
    #             # bboxes = tf.expand_dims(bboxes, axis=0)
    #             scores = tf.gather(bbox_probs[i], indices)
    #             selected_indices = tf.image.non_max_suppression(
    #                 bboxes, scores,
    #                 max_output_size=10,
    #                 iou_threshold=0.5)
    #             bboxes = tf.gather(bboxes, selected_indices)
    #             out_img = tf.py_func(vis.visualize_bboxes_on_image,
    #                                  [img, bboxes], tf.uint8)
    #             return tf.expand_dims(out_img, axis=0)
    #             # return tf.image.draw_bounding_boxes(
    #             #    images[i], bboxes)
    #
    #         out_image = tf.cond(
    #             tf.greater(tf.rank(indices), 0),
    #             true_fn=_draw_bboxes,
    #             false_fn=lambda: images[i])
    #         out_images.append(out_image)

    def create_placeholders(self):
        tf_placeholders = {
            'image': tf.placeholder(
                tf.float32,
                shape=[self.img_h, self.img_w, self.col_channels]),
            'patches': tf.placeholder(
                tf.float32,
                shape=[self.n_rows * self.n_cols,
                       self.patch_h / self.out_stride,
                       self.patch_w / self.out_stride,
                       self.col_channels])
        }
        return tf_placeholders

    def preprocess_graph(self):
        """Creates graph for preprocessing"""
        image = tf.placeholder(
            tf.float32,
            shape=[self.img_h, self.img_w, self.col_channels])
        patches = self.create_patches(image)
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
        bbox_classes = g.get_tensor_by_name('import/bbox_classes:0')
        bbox_regs = g.get_tensor_by_name('import/bbox_regs:0')
        return {'patches': patches,
                'heatmaps': heatmaps,
                'vecmaps': vecmaps,
                'bbox_classes': bbox_classes,
                'bbox_regs': bbox_regs}

    def postprocess_graph(self):
        """Creates graph for post processing"""
        heatmaps = tf.placeholder(
            tf.float32,
            shape=[self.n_rows * self.n_cols,
                   self.patch_h / self.out_stride,
                   self.patch_w / self.out_stride,
                   self.col_channels])
        keypoints = self.stitch_patches(heatmaps)
        return {'heatmaps': heatmaps,
                'keypoints': keypoints}

    def _run_inference(self, sess, image):
        t0 = time.time()
        patches = sess.run(
            self.preprocess_tensors['patches'],
            feed_dict={self.preprocess_tensors['image']: image})
        t1 = time.time()

        heatmaps = sess.run(
            self.network_tensors['heatmaps'],
            feed_dict={self.network_tensors['patches']: patches})
        t2 = time.time()

        out = sess.run(
            self.postprocess_tensors['keypoints'],
            feed_dict={self.postprocess_tensors['heatmaps']: heatmaps})
        # some additional post-processing
        threshold = 0.5
        out[out > threshold] = 1.
        out[out < threshold] = 0.
        out = (255. * out).astype(np.uint8)
        t3 = time.time()
        return out, [t0, t1, t2, t3]

    def display_output(self, image, out):
        out = cv2.resize(out, (self.img_w, self.img_h))
        # back to BGR for opencv display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out = cv2.addWeighted(image, 0.4, out, 0.6, 0)
        cv2.imshow('out', out)
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
            for img_file in img_files:
                image = cv2.imread(img_file)
                image = self.preprocess_image(image)
                out, t = self._run_inference(sess, image)
                stats.update(t)
                if not self.display_output(image, out):
                    break
        elif input_type == 'video':
            video_file = self.infer_cfg.video
            cap = cv2.VideoCapture(video_file)
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break
                image = self.preprocess_image(image)
                out, t = self._run_inference(sess, image)
                stats.update(t)
                if not self.display_output(image, out):
                    break
            cap.release()
        elif input_type == 'camera':
            pass
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
