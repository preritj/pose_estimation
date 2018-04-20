"""
Pose estimation using mobilenet v2
"""

import collections
import tensorflow as tf
from nets import mobilenet_v2 as mobilenet
from utils import ops
from model import Model

slim = tf.contrib.slim


class MobilenetPose(Model):
    def __init(self, cfg):
        self._depth_multiplier = cfg.depth_multiplier
        self._min_depth = cfg.min_depth
        self._skip_layers = cfg.skip_layers
        self._fpn_depth = cfg.fpn_depth
        self._num_keypoints = cfg.num_keypoints
        super().__init__()

    def get_output_shape(self):
        h, w = self._input_shape
        return h / 4, w / 4, self._num_keypoints

    def preprocess(self, inputs):
        """Image preprocessing"""
        return (2.0 / 255.0) * inputs - 1.0

    def build_net(self, preprocessed_inputs):
        image_features = self.encoder(preprocessed_inputs)
        out = self.decoder(image_features)
        return out

    def encoder(self, preprocessed_inputs, scope=None):
        with tf.variable_scope(scope, 'encoder', preprocessed_inputs):
            with slim.arg_scope(mobilenet.mobilenet_v2_arg_scope(
                is_training=self._is_training)
            ):
                _, image_features = mobilenet.mobilenet_v2_base(
                    preprocessed_inputs,
                    final_endpoint=self._skip_layers[-1],
                    min_depth=self._min_depth,
                    depth_multiplier=self._depth_multiplier,
                    scope=scope)
        return {l: image_features[l] for l in self._skip_layers}

    def decoder(self, image_features, scope=None):
        """Builds decoder
        Args:
          image_features: list of image feature tensors to be used for
            skip connections
          scope: A scope name to wrap this op under.
        Returns:
          feature_maps: an OrderedDict mapping keys (feature map names) to
            tensors where each tensor has shape [batch, height_i, width_i, depth_i].
        """
        with tf.variable_scope(scope, 'decoder', image_features):
            with slim.arg_scope(mobilenet.mobilenet_v2_arg_scope(
                    is_training=self._is_training)
            ):
                n_skips = len(self._skip_layers)
                fpn_layers = {}
                last_layer = self._skip_layers[-1]
                net = image_features[last_layer]
                for i in range(n_skips - 2, 0, -1):
                    fpn_name = 'InvertedResidual_fpn_' + str(i + 1)
                    net = mobilenet.inverted_residual_bottleneck(
                        net,
                        depth=self._fpn_depth,
                        stride=1,
                        expand_ratio=6,
                        scope=fpn_name)
                    fpn_layers[fpn_name] = net
                    net = ops.nearest_neighbor_upsampling(net, 2)
                    skip_layer = self._skip_layers[i]
                    net = tf.concat([net, image_features[skip_layer]], -1)
                net = mobilenet.inverted_residual_bottleneck(
                    net,
                    depth=64,
                    stride=1,
                    expand_ratio=6,
                    scope='InvertedResidual_final')
                net = slim.conv2d(net, self._num_keypoints, [1, 1],
                                  activation_fn=None,
                                  scope='heatmap')
        return net
