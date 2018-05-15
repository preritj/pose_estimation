"""
Base class for pose estimation
"""
from abc import abstractmethod
import tensorflow as tf


EPSILON = 1e-5


class Model:
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self._num_keypoints = self.cfg.num_keypoints
        self._num_vecs = self.cfg.num_vecs
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
        heatmaps, vecmaps, fpn_features = self.build_net(
            preprocessed_inputs, is_training=is_training)
        bbox_clf_logits, bbox_regs = self.bbox_clf_reg_net(
            fpn_features, is_training)
        prediction = {'heatmaps': heatmaps,
                      'vecmaps': vecmaps,
                      'bbox_clf_logits': bbox_clf_logits,
                      'bbox_regs': bbox_regs}
        return prediction

    def heatmap_loss(self, labels, logits, weights):
        logits = tf.reshape(logits, [-1, self._num_keypoints])
        labels = tf.reshape(labels, [-1, self._num_keypoints])
        weights = tf.reshape(weights, [-1])
        heatmap_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=logits
        )
        # heatmap_loss = tf.losses.sigmoid_cross_entropy(
        #     multi_class_labels=labels,
        #     logits=logits,
        #     reduction=tf.losses.Reduction.NONE)
        heatmap_loss = tf.reduce_sum(heatmap_loss, axis=-1)
        heatmap_loss = tf.reduce_mean(heatmap_loss * weights)
        return heatmap_loss

    def vecmap_loss(self, regs_gt, regs_pred):
        weights = tf.cast(tf.greater(tf.abs(regs_gt), EPSILON),
                          tf.float32)
        weights /= tf.maximum(EPSILON, tf.abs(regs_gt))
        vecmap_loss = tf.losses.absolute_difference(
            labels=regs_gt,
            predictions=regs_pred,
            weights=weights)
        return vecmap_loss

    def bbox_clf_reg_loss(self, clf_labels, clf_logits,
                          clf_weights, regs_gt, regs_pred):
        clf_labels = tf.reshape(clf_labels, [-1])
        clf_weights = tf.reshape(clf_weights, [-1])
        # n_pos_labels = tf.to_float(tf.reduce_sum(clf_labels))
        # n_labels = tf.reduce_sum(clf_weights)
        # n_neg_labels = n_labels - n_pos_labels
        # scale_factor = tf.cond(
        #     tf.greater(n_pos_labels, 0),
        #     lambda: tf.pow(n_neg_labels / n_pos_labels, .3),
        #     lambda: 1.)
        # clf_weights += tf.to_float(clf_labels) * scale_factor
        regs_gt = tf.reshape(regs_gt, [-1, 4])
        clf_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=clf_labels,
            logits=clf_logits,
            weights=clf_weights)
        reg_loss = tf.losses.huber_loss(
            labels=regs_gt,
            predictions=regs_pred,
            delta=1.,
            reduction=tf.losses.Reduction.NONE)
        # reg_loss = tf.losses.mean_squared_error(
        #     labels=regs_gt,
        #     predictions=regs_pred,
        #     reduction=tf.losses.Reduction.NONE)
        reg_loss = tf.reduce_sum(reg_loss, axis=-1)
        reg_loss = tf.reduce_mean(
            reg_loss * tf.to_float(clf_labels))
        return clf_loss, reg_loss

    def losses(self, prediction, ground_truth):
        heatmaps_logits = prediction['heatmaps']
        vecmaps_regs = prediction['vecmaps']
        bbox_clf_logits = prediction['bbox_clf_logits']
        bbox_regs_pred = prediction['bbox_regs']
        heatmaps_gt = ground_truth['heatmaps']
        vecmaps_regs_gt = ground_truth['vecmaps']
        bbox_clf_gt = ground_truth['bboxes']['classes']
        bbox_regs_gt = ground_truth['bboxes']['regs']
        bbox_weights = ground_truth['bboxes']['weights']
        mask_weights = ground_truth['masks']

        heatmap_loss = self.heatmap_loss(
            labels=heatmaps_gt,
            logits=heatmaps_logits,
            weights=mask_weights)

        vecmap_loss = self.vecmap_loss(
            regs_gt=vecmaps_regs_gt,
            regs_pred=vecmaps_regs
        )

        bbox_clf_loss, bbox_reg_loss = self.bbox_clf_reg_loss(
            clf_labels=bbox_clf_gt,
            clf_logits=bbox_clf_logits,
            clf_weights=bbox_weights,
            regs_gt=bbox_regs_gt,
            regs_pred=bbox_regs_pred)

        losses = {'heatmap_loss': heatmap_loss,
                  'vecmap_loss': vecmap_loss,
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
