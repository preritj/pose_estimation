"""
Base class for pose estimation
"""
from abc import abstractmethod
import tensorflow as tf


class Model:
    def __init__(self):
        self._is_training = True
        self._batch_size = 32
        self._input_shape = (360, 360)
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
                               shape=(batch_size, in_h, in_w),
                               name='masks')
        return {'images': images, 'heatmaps': heatmaps}

    @abstractmethod
    def preprocess(self, inputs):
        """Image preprocessing"""
        pass

    @abstractmethod
    def build_net(self, preprocessed_inputs):
        """Builds network and returns heatmap logits"""
        pass

    @staticmethod
    def resize_masks(masks, out_size):


    def make_train_op(self):
        images = self.tf_placeholders['images']
        heatmaps = self.tf_placeholders['heatmaps']
        masks = self.tf_placeholders['masks']
        images = self.preprocess(images)
        heatmaps_pred = self.build_net(images)
        loss = tf.nn.l2_loss(heatmaps, )
        pass

