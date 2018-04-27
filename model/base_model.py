"""
Base class for pose estimation
"""
from abc import abstractmethod
import tensorflow as tf


class Model:
    def __init__(self, model_cfg):
        self.cfg = model_cfg
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

    @staticmethod
    def losses(prediction, ground_truth):
        heatmaps_pred = prediction['heatmaps']
        heatmaps_gt = ground_truth['heatmaps']
        weights = ground_truth['masks']
        l2_loss = tf.losses.mean_squared_error(
            heatmaps_gt, heatmaps_pred,
            reduction=tf.losses.Reduction.NONE)
        l2_loss = weights * tf.reduce_mean(l2_loss, axis=-1)
        l2_loss = tf.reduce_mean(l2_loss)
        # TODO : add regularization losses
        losses = {'l2_loss': l2_loss}
        return losses
