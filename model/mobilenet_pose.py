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
    def __init(self):
        self._depth_multiplier = 1.0
        self._min_depth = 8
        self._skip_layers = ['InvertedResidual_32_2',
                             'InvertedResidual_64_3',
                             'InvertedResidual_160_2']
        self._fpn_depth = 160
        super().__init__()

    def preprocess(self, inputs):
        """Image preprocessing"""
        return (2.0 / 255.0) * inputs - 1.0

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
                fpn_layers = ['fpn_'+str(i) for i in range(n_skips)]
                last_layer = self._skip_layers[-1]
                net = image_features[last_layer]
                for skip_layer, fpn_layer in zip(
                        self._skip_layers[::-1], fpn_layers[::-1]):
                    net = mobilenet.inverted_residual_bottleneck(
                        net,
                        depth=self._fpn_depth,
                        stride=1,
                        expand_ratio=6,
                        scope=fpn_layer)
                    net = ops.nearest_neighbor_upsampling(net, 2)
                    net = tf.concat([net, image_features[skip_layer]], -1)
                # TODO : build the final mask head
