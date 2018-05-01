"""
Base class for pose estimation
"""
from abc import abstractmethod
import tensorflow as tf


class Model:
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self._num_keypoints = self.cfg.num_keypoints
        self.check_output_shape()

    @abstractmethod
    def check_output_shape(self):
        """Check shape consistency"""
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def preprocess(self, inputs):
        """Image preprocessing"""
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def build_net(self, preprocessed_inputs, is_training=False):
        """Builds network and returns heatmaps and fpn features"""
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def bbox_clf_reg_net(self, fpn_features, is_training=False):
        """Builds bbox classifier and regressor"""
        raise NotImplementedError("Not yet implemented")

    def predict(self, inputs, is_training=False):
        images = inputs['images']
        preprocessed_inputs = self.preprocess(images)
        heatmaps, fpn_features = self.build_net(
            preprocessed_inputs, is_training=is_training)
        bbox_clf_logits, bbox_regs = self.bbox_clf_reg_net(
            fpn_features, is_training)
        prediction = {'heatmaps': heatmaps,
                      'bbox_clf_logits': bbox_clf_logits,
                      'bbox_regs': bbox_regs}
        return prediction

    def heatmap_loss(self, labels, logits, weights):
        logits = tf.reshape(logits, [-1, self._num_keypoints])
        labels = tf.reshape(labels, [-1, self._num_keypoints])
        weights = tf.reshape(weights, [-1])
        heatmap_loss = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=labels,
            logits=logits,
            reduction=tf.losses.Reduction.NONE)
        heatmap_loss = tf.reduce_sum(heatmap_loss, axis=-1)
        heatmap_loss = tf.reduce_mean(heatmap_loss * weights)
        return heatmap_loss

    def bbox_clf_reg_loss(self, clf_labels, clf_logits,
                          clf_weights, regs_gt, regs_pred):
        clf_labels = tf.reshape(clf_labels, [-1])
        clf_weights = tf.reshape(clf_weights, [-1])
        regs_gt = tf.reshape(regs_gt, [-1, 4])
        clf_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=clf_labels,
            logits=clf_logits,
            weights=clf_weights)
        reg_loss = tf.losses.mean_squared_error(
            labels=regs_gt,
            predictions=regs_pred,
            reduction=tf.losses.Reduction.NONE)
        reg_loss = tf.reduce_sum(reg_loss, axis=-1)
        reg_loss = tf.reduce_mean(
            reg_loss * tf.to_float(clf_labels))
        return clf_loss, reg_loss

    def losses(self, prediction, ground_truth):
        heatmaps_logits = prediction['heatmaps']
        bbox_clf_logits = prediction['bbox_clf_logits']
        bbox_regs_pred = prediction['bbox_regs']
        heatmaps_gt = ground_truth['heatmaps']
        bbox_clf_gt = ground_truth['bboxes']['classes']
        bbox_regs_gt = ground_truth['bboxes']['regs']
        bbox_weights = ground_truth['bboxes']['weights']
        mask_weights = ground_truth['masks']

        heatmap_loss = self.heatmap_loss(
            labels=heatmaps_gt,
            logits=heatmaps_logits,
            weights=mask_weights)

        bbox_clf_loss, bbox_reg_loss = self.bbox_clf_reg_loss(
            clf_labels=bbox_clf_gt,
            clf_logits=bbox_clf_logits,
            clf_weights=bbox_weights,
            regs_gt=bbox_regs_gt,
            regs_pred=bbox_regs_pred)

        losses = {'heatmap_loss': heatmap_loss,
                  'bbox_clf_loss': bbox_clf_loss,
                  'bbox_reg_loss': bbox_reg_loss}
        # l2_loss = tf.losses.mean_squared_error(
        #     heatmaps_gt, heatmaps_pred,
        #     reduction=tf.losses.Reduction.NONE)
        # l2_loss = weights * tf.reduce_mean(l2_loss, axis=-1)
        # l2_loss = tf.reduce_mean(l2_loss)
        # # TODO : add regularization losses
        # losses = {'l2_loss': l2_loss}
        return losses
