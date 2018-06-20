"""
Pose estimation using mobilenet v2
"""
import collections
import tensorflow as tf
from nets.mobilenet_v2 import mobilenet_base, training_scope
from nets.conv_blocks import expanded_conv
from utils import ops
from model.base_model import Model

slim = tf.contrib.slim


class MobilenetPose(Model):
    def __init__(self, cfg):
        self._num_classes = 2
        self._depth_multiplier = cfg.depth_multiplier
        self._min_depth = cfg.min_depth
        super().__init__(cfg)

    def check_output_shape(self):
        h, w = self.cfg.input_shape
        assert (h % 32 == 0) and (w % 32 == 0), \
            "input dimensions must be multiples of 32"
        out_h, out_w = h / 8, w / 8
        assert [out_h, out_w] == list(self.cfg.output_shape), \
            "output_shape inconsistent with model output shape"
        return h / 8, w / 8, self._num_keypoints

    def preprocess(self, inputs):
        """Image preprocessing"""
        return (2.0 / 255.0) * inputs - 1.0

    def build_net(self, preprocessed_inputs, is_training=False):
        features = self.backbone(preprocessed_inputs, is_training)
        out = self.head(features, is_training)
        return out

    def backbone(self, preprocessed_inputs, is_training=False, scope=None):
        """Builds the mobilenet backbone"""
        with tf.variable_scope(scope, 'backbone'):
            with slim.arg_scope(training_scope(
                is_training=is_training)
            ):
                features, _ = mobilenet_base(
                    preprocessed_inputs,
                    final_endpoint=self.cfg.backbone_endpoint,
                    min_depth=self._min_depth,
                    depth_multiplier=self._depth_multiplier,
                    output_stride=self.cfg.output_stride,
                    scope=scope)
        return features

    def head(self, features, is_training=False, scope=None):
        """Builds the network head
        Args:
          features: output of backbone
          is_training (bool) : true if training mode
          scope: A scope name to wrap this op under.
        Returns:
          feature_maps: an OrderedDict mapping keys (feature map names) to
            tensors where each tensor has shape [batch, height_i, width_i, depth_i].
        """
        with tf.variable_scope(scope, 'head'):
            with slim.arg_scope(training_scope(
                    is_training=is_training)
            ):
                net = features
                heatmap_branch = expanded_conv(
                    net,
                    num_outputs=self.cfg.final_depth,
                    stride=1,
                    scope='heatmap_branch')
                vecmap_branch = expanded_conv(
                    net,
                    num_outputs=self.cfg.final_depth,
                    stride=1,
                    scope='vecmap_branch')
                offsetmap_branch = expanded_conv(
                    net,
                    num_outputs=self.cfg.final_depth,
                    stride=1,
                    scope='offsetmap_branch')
            heatmap = slim.conv2d(heatmap_branch, self._num_keypoints, [1, 1],
                                  activation_fn=None, normalizer_fn=None,
                                  normalizer_params=None, scope='heatmap')
            vecmap = slim.conv2d(vecmap_branch, self._num_vecs, [1, 1],
                                 activation_fn=None, normalizer_fn=None,
                                 normalizer_params=None, scope='vecmap')
            offsetmap = slim.conv2d(offsetmap_branch, 2 * self._num_keypoints, [1, 1],
                                    activation_fn=None, normalizer_fn=None,
                                    normalizer_params=None, scope='offsetmap')
        return heatmap, vecmap, offsetmap

