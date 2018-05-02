import tensorflow as tf
from utils.bboxes import bbox_decode


def tf_vis(features, predictions, train_cfg, model_cfg):
    images = features['images']
    heatmaps_logits = predictions['heatmaps']
    heatmaps = tf.nn.sigmoid(heatmaps_logits)
    heatmap_out = tf.expand_dims(
        tf.zeros_like(heatmaps[:, :, :, 0]), axis=-1)
    heatmap_out = tf.concat([heatmap_out, heatmaps], -1)
    bbox_clf_logits = predictions['bbox_clf_logits']
    _, bbox_probs = tf.split(
        tf.nn.softmax(bbox_clf_logits),
        num_or_size_splits=2,
        axis=1)
    bbox_probs = tf.squeeze(bbox_probs)
    bbox_regs = predictions['bbox_regs']
    bbox_regs = bbox_decode(bbox_regs, anchors, )
    batch_size = train_cfg.batch_size
    # create a list of anchors for clf heads
    num_anchors = []
    num_anchors_per_grid_cell = (
        len(model_cfg.anchor_scales)
        * len(model_cfg.anchor_ratios))
    input_shape = model_cfg.input_shape
    n_clf_heads = len(model_cfg.anchor_strides)
    grid_shapes = []
    for stride in model_cfg.anchor_strides:
        grid_h = input_shape[0] / stride
        grid_w = input_shape[1] / stride
        grid_shapes.append([grid_h, grid_w])
        num_anchors.append(
            batch_size * grid_h * grid_w
            * num_anchors_per_grid_cell)
    bbox_probs = tf.split(
        bbox_probs, num_or_size_splits=num_anchors,
        axis=0)
    for i in range(n_clf_heads):
        grid_h, grid_w = grid_shapes[i]
        bbox_probs[i] = tf.reshape(
            bbox_probs[i],
            [batch_size, grid_h, grid_w, num_anchors_per_grid_cell]
        )
    bbox_regs_pred = predictions['bbox_regs']