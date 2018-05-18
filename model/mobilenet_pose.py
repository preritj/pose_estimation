"""
Pose estimation using mobilenet v2
"""
import collections
import tensorflow as tf
from nets.mobilenet_v2 import (
    mobilenet_v2_base, mobilenet_v2_arg_scope,
    inverted_residual_bottleneck)
from utils import ops
from model.base_model import Model

slim = tf.contrib.slim


class MobilenetPose(Model):
    def __init__(self, cfg):
        self._num_classes = 2
        self._depth_multiplier = cfg.depth_multiplier
        self._min_depth = cfg.min_depth
        self._skip_layers = cfg.skip_layers
        self._fpn_depth = cfg.fpn_depth
        self._boxes_per_anchor = (len(cfg.anchor_scales) *
                                  len(cfg.anchor_ratios))
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
        image_features = self.encoder(preprocessed_inputs)
        out = self.decoder(image_features, is_training)
        return out

    def encoder(self, preprocessed_inputs, is_training=False, scope=None):
        with tf.variable_scope(scope, 'encoder'):
            with slim.arg_scope(mobilenet_v2_arg_scope(
                is_training=is_training)
            ):
                _, image_features = mobilenet_v2_base(
                    preprocessed_inputs,
                    final_endpoint=self._skip_layers[-1],
                    min_depth=self._min_depth,
                    depth_multiplier=self._depth_multiplier,
                    scope=scope)
        return {l: image_features[l] for l in self._skip_layers}

    def decoder(self, image_features, is_training=False, scope=None):
        """Builds decoder
        Args:
          image_features: dict of image feature tensors to be used for
            skip connections
          is_training (bool) : true if training mode
          scope: A scope name to wrap this op under.
        Returns:
          feature_maps: an OrderedDict mapping keys (feature map names) to
            tensors where each tensor has shape [batch, height_i, width_i, depth_i].
        """
        with tf.variable_scope(scope, 'decoder'):
            with slim.arg_scope(mobilenet_v2_arg_scope(
                    is_training=is_training)
            ):
                n_skips = len(self._skip_layers)
                fpn_layers = collections.OrderedDict()
                last_layer = self._skip_layers[-1]
                net = image_features[last_layer]
                for i in range(n_skips - 2, -1, -1):
                    fpn_name = 'InvertedResidual_fpn_' + str(i + 1)
                    net = inverted_residual_bottleneck(
                        net,
                        depth=self._fpn_depth,
                        stride=1,
                        expand_ratio=6,
                        scope=fpn_name)
                    fpn_layers[fpn_name] = net
                    if self.cfg.upsampling == 'tf':
                        net = ops.nearest_neighbor_upsampling(net, 2)
                    else:
                        net = ops.upsample(net)
                    skip_layer = self._skip_layers[i]
                    net = tf.concat([net, image_features[skip_layer]], -1)
                heatmap_branch = inverted_residual_bottleneck(
                    net,
                    depth=64,
                    stride=1,
                    expand_ratio=6,
                    scope='InvertedResidual_heatmap')
                vecmap_branch = inverted_residual_bottleneck(
                    net,
                    depth=64,
                    stride=1,
                    expand_ratio=6,
                    scope='InvertedResidual_vecmap')
            heatmap = slim.conv2d(heatmap_branch, self._num_keypoints, [1, 1],
                                  activation_fn=None, normalizer_fn=None,
                                  normalizer_params=None, scope='heatmap')
            vecmap = slim.conv2d(vecmap_branch, self._num_vecs, [1, 1],
                                 activation_fn=None, normalizer_fn=None,
                                 normalizer_params=None, scope='vecmap')
        return heatmap, vecmap, fpn_layers

    def bbox_clf_reg_net(self, fpn_features, is_training=False, scope=None):
        """Builds bbox classifier and regressor"""
        num_fpn_layers = len(self.cfg.base_anchor_sizes)
        assert num_fpn_layers == len(fpn_features), \
            "Number of anchor sizes must match number of fpn layers"
        bbox_clf_logits = self.bbox_clf_net(fpn_features)
        bbox_regs = self.bbox_reg_net(fpn_features)
        return bbox_clf_logits, bbox_regs

    def bbox_clf_net(self, fpn_features):
        """Builds bbox classifier
        Args:
          fpn_features : dictionary of FPN features with keys as layer name
            and values as features (all features must have same depth)
            Features have shape [N, h_i, w_i, c_i]
          is_training (bool) : true if training mode
          scope: A scope name to wrap this op under.
        Returns:
            bbox clf logits : A list of tensors where each tensor in the list
            corresponds to an fpn layer in the input fpn_features dictionary.
            Each tensor has shape [N, h_i, w_i, K * (num of boxes per anchor)]
            where K is the number of classes
            """
        bbox_clf_logits = []
        with tf.variable_scope('bbox_clf', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d], activation_fn=None,
                                normalizer_fn=None, normalizer_params=None):
                strides = self.cfg.anchor_strides
                for stride, fpn_name in zip(strides,
                                            reversed(fpn_features.keys())):
                    net = fpn_features[fpn_name]
                    # net = inverted_residual_bottleneck(
                    #     net,
                    #     depth=64,
                    #     stride=1,
                    #     expand_ratio=6,
                    #     scope=fpn_name)
                    net = slim.conv2d(
                        net, self._fpn_depth, [3, 3], scope='clf_rpn')
                    net = slim.conv2d(
                        net, self._num_classes * self._boxes_per_anchor,
                        [1, 1], scope='clf_feat')
                    grid_h = self.cfg.input_shape[0] // stride
                    grid_w = self.cfg.input_shape[1] // stride
                    n_anchors = grid_h * grid_w * self._boxes_per_anchor
                    logits = tf.reshape(
                        net, [-1, n_anchors * self._num_classes])
                    bbox_clf_logits.append(logits)
        bbox_clf_logits = tf.concat(bbox_clf_logits, axis=1)
        bbox_clf_logits = tf.reshape(
            bbox_clf_logits, [-1, self._num_classes])
        return bbox_clf_logits

    def bbox_reg_net(self, fpn_features):
        """Builds bbox regressor
        Args:
          fpn_features : dictionary of FPN features with keys as layer name
            and values as features (all features must have same depth)
            Features have shape [N, h_i, w_i, c_i]
          is_training (bool) : true if training mode
          scope: A scope name to wrap this op under.
        Returns:
            bbox regressions : A list of tensors where each tensor in the list
            corresponds to an fpn layer in the input fpn_features dictionary.
            Each tensor has shape [N, h_i, w_i, 4 * (num of boxes per anchor)]
            """
        bbox_regs = []
        with tf.variable_scope('bbox_reg', reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d], activation_fn=None,
                                normalizer_fn=None, normalizer_params=None):
                strides = self.cfg.anchor_strides
                for stride, fpn_name in zip(strides,
                                            reversed(fpn_features.keys())):
                    net = fpn_features[fpn_name]
                    # net = inverted_residual_bottleneck(
                    #     net,
                    #     depth=64,
                    #     stride=1,
                    #     expand_ratio=6,
                    #     scope=fpn_name)
                    net = slim.conv2d(
                        net, self._fpn_depth, [3, 3], scope='reg_rpn')
                    net = slim.conv2d(
                        net, 4 * self._boxes_per_anchor,
                        [1, 1], scope='reg_feat')
                    grid_h = self.cfg.input_shape[0] // stride
                    grid_w = self.cfg.input_shape[1] // stride
                    n_anchors = grid_h * grid_w * self._boxes_per_anchor
                    regs = tf.reshape(net, [-1, 4 * n_anchors])
                    bbox_regs.append(regs)
        bbox_regs = tf.concat(bbox_regs, axis=1)
        bbox_regs = tf.reshape(bbox_regs, [-1, 4])
        return bbox_regs
