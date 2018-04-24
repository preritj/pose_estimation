"""
Pose estimation using mobilenet v2
"""

import tensorflow as tf
from nets import mobilenet_v2 as mobilenet
from utils import ops
from model.base_model import Model

slim = tf.contrib.slim


class MobilenetPose(Model):
    def __init(self, cfg):
        self._depth_multiplier = cfg.depth_multiplier
        self._min_depth = cfg.min_depth
        self._skip_layers = cfg.skip_layers
        self._fpn_depth = cfg.fpn_depth
        self._num_keypoints = cfg.num_keypoints
        self.check_output_shape()
        super().__init__(cfg)

    def check_output_shape(self):
        h, w = self.cfg.input_shape
        out_h, out_w = h / 4, w / 4
        assert ([out_h, out_w] == list(self.cfg.output_shape),
                "output_shape inconsistent with model output shape")
        return h / 4, w / 4, self._num_keypoints

    def _preprocess(self, inputs):
        """Image preprocessing"""
        return (2.0 / 255.0) * inputs - 1.0

    def _build_net(self, preprocessed_inputs, is_training=False):
        image_features = self._encoder(preprocessed_inputs)
        out = self._decoder(image_features, is_training)
        return out

    def _encoder(self, preprocessed_inputs, is_training=False, scope=None):
        with tf.variable_scope(scope, 'encoder', preprocessed_inputs):
            with slim.arg_scope(mobilenet.mobilenet_v2_arg_scope(
                is_training=is_training)
            ):
                _, image_features = mobilenet.mobilenet_v2_base(
                    preprocessed_inputs,
                    final_endpoint=self._skip_layers[-1],
                    min_depth=self._min_depth,
                    depth_multiplier=self._depth_multiplier,
                    scope=scope)
        return {l: image_features[l] for l in self._skip_layers}

    def _decoder(self, image_features, is_training=False, scope=None):
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
                    is_training=is_training)
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
