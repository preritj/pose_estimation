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

    def predict(self, inputs, is_training=False):
        images = inputs['images']
        preprocessed_inputs = self.preprocess(images)
        heatmaps, vecmaps, offsetmaps = self.build_net(
            preprocessed_inputs, is_training=is_training)
        prediction = {'heatmaps': heatmaps,
                      'vecmaps': vecmaps,
                      'offsetmaps': offsetmaps}
        return prediction

    def heatmap_loss(self, labels, logits, weights):
        logits = tf.reshape(logits, [-1, self._num_keypoints])
        labels = tf.reshape(labels, [-1, self._num_keypoints])
        weights = tf.reshape(weights, [-1, self._num_keypoints])
        heatmap_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels,
            logits=logits
        )
        # heatmap_loss = tf.reduce_sum(heatmap_loss, axis=-1)
        heatmap_loss = tf.reduce_mean(heatmap_loss * weights)
        return heatmap_loss

    def vecmap_loss(self, regs_gt, regs_pred):
        weights = tf.cast(tf.greater(tf.abs(regs_gt), EPSILON),
                          tf.float32)
        # weights /= tf.maximum(EPSILON, tf.abs(regs_gt))
        vecmap_loss = tf.losses.absolute_difference(
            labels=regs_gt,
            predictions=regs_pred,
            weights=weights)
        return vecmap_loss

    def offsetmap_loss(self, regs_gt, regs_pred, heatmaps_gt,
                       heatmaps_weights):
        weights = tf.cast(tf.greater(heatmaps_gt, EPSILON), tf.float32)
        weights = weights * heatmaps_weights
        # weights = tf.expand_dims(weights, -1)
        # weights = tf.tile(weights, [1, 1, 1, 1, 2])
        # weights = tf.reshape(weights, tf.shape(regs_gt))
        offsetmap_loss = tf.losses.absolute_difference(
            labels=regs_gt,
            predictions=regs_pred,
            reduction=tf.losses.Reduction.NONE)
        offsetmap_loss = tf.reshape(offsetmap_loss, [-1, 2])
        offsetmap_loss = tf.reduce_sum(offsetmap_loss, axis=1)
        offsetmap_loss = offsetmap_loss * tf.reshape(weights, [-1])
        offsetmap_loss = tf.reduce_mean(offsetmap_loss)
        return offsetmap_loss

    def losses(self, prediction, ground_truth):
        heatmaps_logits = prediction['heatmaps']
        vecmaps_regs = prediction['vecmaps']
        offsetmaps_regs = prediction['offsetmaps']
        heatmaps_gt = ground_truth['heatmaps']
        vecmaps_regs_gt = ground_truth['vecmaps']
        offsetmaps_regs_gt = ground_truth['offsetmaps']
        mask_weights = ground_truth['masks']

        heatmap_loss = self.heatmap_loss(
            labels=heatmaps_gt,
            logits=heatmaps_logits,
            weights=mask_weights)

        vecmap_loss = self.vecmap_loss(
            regs_gt=vecmaps_regs_gt,
            regs_pred=vecmaps_regs)

        offsetmap_loss = self.offsetmap_loss(
            regs_gt=offsetmaps_regs_gt,
            regs_pred=offsetmaps_regs,
            heatmaps_gt=heatmaps_gt,
            heatmaps_weights=mask_weights)

        losses = {'heatmap_loss': heatmap_loss,
                  'vecmap_loss': vecmap_loss,
                  'offsetmap_loss': offsetmap_loss}
        # l2_loss = tf.losses.mean_squared_error(
        #     heatmaps_gt, heatmaps_pred,
        #     reduction=tf.losses.Reduction.NONE)
        # l2_loss = weights * tf.reduce_mean(l2_loss, axis=-1)
        # l2_loss = tf.reduce_mean(l2_loss)
        # # TODO : add regularization losses
        # losses = {'l2_loss': l2_loss}
        return losses
