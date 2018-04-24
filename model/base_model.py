"""
Base class for pose estimation
"""
from abc import abstractmethod
import tensorflow as tf


class Model:
    def __init__(self, model_cfg):
        self.cfg = model_cfg

    @abstractmethod
    def _preprocess(self, inputs):
        """Image preprocessing"""
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def _build_net(self, preprocessed_inputs, is_training=False):
        """Builds network and returns heatmaps"""
        raise NotImplementedError("Not yet implemented")

    def predict(self, inputs, is_training=False):
        images = inputs['images']
        preprocessed_inputs = self._preprocess(images)
        heatmaps = self._build_net(preprocessed_inputs,
                                   is_training=is_training)
        prediction = {'heatmaps': heatmaps}
        return prediction

    @staticmethod
    def losses(prediction, ground_truth):
        heatmaps_pred = prediction['heatmaps']
        heatmaps_gt = ground_truth['heatmaps']
        weights = ground_truth['masks']
        l2_loss = tf.losses.mean_squared_error(
            heatmaps_gt, heatmaps_pred, weights=weights,
            reduction=tf.losses.Reduction.SUM)
        # TODO : add regularization losses
        losses = {'l2_loss': l2_loss}
        return losses
