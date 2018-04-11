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
        self._feature_layers = ['InvertedResidual_32_2',
                                'InvertedResidual_64_3',
                                'InvertedResidual_160_2']
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
                    final_endpoint=self._feature_layers[-1],
                    min_depth=self._min_depth,
                    depth_multiplier=self._depth_multiplier,
                    scope=scope)
        return image_features

    def decoder(self, image_features, depth=160, scope=None):
        """Builds decoder
        Args:
          image_features: list of image feature tensors to be used for
            skip connections
          depth: depth of output feature maps.
          scope: A scope name to wrap this op under.
        Returns:
          feature_maps: an OrderedDict mapping keys (feature map names) to
            tensors where each tensor has shape [batch, height_i, width_i, depth_i].
        """
        with tf.variable_scope(scope, 'decoder', image_features):
            num_levels = len(image_features)
            output_feature_maps_list = []
            output_feature_map_keys = []
            with slim.arg_scope(
                    [slim.conv2d, slim.separable_conv2d], padding='SAME'):
                top_down = slim.conv2d(
                    image_features[-1],
                    depth, [1, 1], scope='projection_%d' % num_levels)
                output_feature_maps_list.append(top_down)
                output_feature_map_keys.append(
                    'top_down_feature_map_%d' % (num_levels - 1))

                for level in reversed(range(num_levels - 1)):
                    top_down = ops.nearest_neighbor_upsampling(top_down, 2)
                    feature_map = mobilenet.inverted_residual_bottleneck(net, depth(conv_def.depth), stride, conv_def.t, scope=end_point)
                    residual = slim.conv2d(
                        image_features[level], depth, [1, 1],
                        scope='projection_%d' % (level + 1))
                    top_down = 0.5 * top_down + 0.5 * residual
                    output_feature_maps_list.append(slim.conv2d(
                        top_down,
                        depth, [3, 3],
                        activation_fn=None,
                        scope='smoothing_%d' % (level + 1)))
                    output_feature_map_keys.append('top_down_feature_map_%d' % level)
                return collections.OrderedDict(
                    reversed(list(zip(output_feature_map_keys,
                                      output_feature_maps_list))))
