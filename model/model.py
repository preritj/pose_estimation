"""
Base class for pose estimation
"""
from abc import abstractmethod
import tensorflow as tf


class Model:
    def __init__(self, train_cfg, model_cfg):
        self.cfg = model_cfg
        self._is_training = train_cfg.is_training
        self._batch_size = train_cfg.batch_size
        self._input_shape =
        self._output_shape = self.get_output_shape()
        self.tf_placeholders = self.create_placeholders()

    @abstractmethod
    def get_output_shape(self):
        """output shape of the heatmaps"""
        pass

    def create_placeholders(self):
        batch_size = self._batch_size
        in_h, in_w = self._input_shape
        out_h, out_w, out_n = self._output_shape
        images = tf.placeholder(tf.float32,
                                shape=(batch_size, in_h, in_w, 3),
                                name='images')
        heatmaps = tf.placeholder(tf.float32,
                                  shape=(batch_size, out_h, out_w, out_n),
                                  name='heatmaps')
        masks = tf.placeholder(tf.float32,
                               shape=(batch_size, out_h, out_w),
                               name='masks')
        return {'images': images,
                'heatmaps': heatmaps,
                'masks': masks}

    @abstractmethod
    def preprocess(self, inputs):
        """Image preprocessing"""
        raise NotImplementedError("Not yet implemented")

    @abstractmethod
    def build_net(self, preprocessed_inputs):
        """Builds network and returns heatmap logits"""
        raise NotImplementedError("Not yet implemented")

    def make_train_op(self):
        images = self.tf_placeholders['images']
        heatmaps = self.tf_placeholders['heatmaps']
        masks = self.tf_placeholders['masks']
        images = self.preprocess(images)
        heatmaps_pred = self.build_net(images)
        l2_loss = tf.losses.mean_squared_error(
            heatmaps, heatmaps_pred, weights=masks,
            reduction=tf.losses.Reduction.SUM)

        solver = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = solver.minimize(loss, global_step=self.global_step)
            self.loss_op = [clf_loss, reg_loss, mask_loss]


