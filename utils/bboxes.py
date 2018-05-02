import tensorflow as tf


EPSILON = 1e-8


def iou(boxes1, boxes2):
    """Compute IoU between boxes.
    Args:
      boxes1: Tensor of shape [N, 4] in format (y1, x1, y2, x2)
      boxes2: Tensor of shape [M, 4] in same format
    Returns:
      a tensor with shape [N, M] representing IoU
    """
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxes1, num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxes2, num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    intersections = intersect_heights * intersect_widths

    areas1 = tf.squeeze((y_max1 - y_min1) * (x_max1 - x_min1), [1])
    areas2 = tf.squeeze((y_max2 - y_min2) * (x_max2 - x_min2), [1])
    unions = (tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0)
              - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections),
        tf.truediv(intersections, unions))


def get_matches(gt_bboxes, pred_bboxes,
                unmatched_threshold=0.4,
                matched_threshold=0.7,
                force_match_for_gt_bbox=True,
                scale_factors=None):

    num_preds = tf.shape(pred_bboxes)[0]

    def _gt_bboxes_absent():
        """returns -1 for all indices since no gt-bbox present"""
        return -1 * tf.ones([num_preds], dtype=tf.int32)

    def _gt_bboxes_present():
        """Returns indices of gt_bboxes assigned to pred_bboxes (or anchors)
          index of -1 means no gt-bbox was assigned because iou was
            below the unmatched-threshold
          index of -2 means no gt-bbox was assigned because iou was
            between unmatched and matched-thresholds"""
        iou_matrix = iou(gt_bboxes, pred_bboxes)
        # assign bbox to each prediction
        matches = tf.argmax(iou_matrix, 0, output_type=tf.int32)
        matched_vals = tf.reduce_max(iou_matrix, 0)
        below_threshold = tf.cast(
            tf.greater(unmatched_threshold, matched_vals),
            tf.int32)
        matches = matches * (1 - below_threshold) - below_threshold
        between_thresholds = tf.logical_and(
            tf.greater_equal(matched_vals, unmatched_threshold),
            tf.greater(matched_threshold, matched_vals))
        between_thresholds = tf.cast(between_thresholds, tf.int32)
        matches = (matches * (1 - between_thresholds)
                   - 2 * between_thresholds)

        if force_match_for_gt_bbox:
            best_pred_matches = tf.argmax(iou_matrix, 1,
                                          output_type=tf.int32)
            best_pred_match_matrix = tf.one_hot(
                best_pred_matches, depth=num_preds)
            force_matches = tf.argmax(
                best_pred_match_matrix, 0, output_type=tf.int32)
            force_matches_mask = tf.cast(
                tf.reduce_max(best_pred_match_matrix, 0), tf.bool)
            matches = tf.where(force_matches_mask,
                               force_matches, matches)
        return matches

    matched_indices = tf.cond(
        tf.equal(tf.shape(gt_bboxes)[0], 0),
        true_fn=_gt_bboxes_absent,
        false_fn=_gt_bboxes_present)

    gather_indices = matched_indices + 2
    dummy_bboxes = tf.zeros([2, 4])
    bboxes = tf.concat([dummy_bboxes, gt_bboxes], axis=0)
    matched_gt_bboxes = tf.gather(bboxes, gather_indices)
    matched_regs = bbox_encode(
        matched_gt_bboxes, pred_bboxes, scale_factors)
    matched_classes = tf.cast(
        tf.greater(matched_indices, -1), tf.int32)
    matched_weights = tf.cast(
        tf.greater(matched_indices, -2), tf.float32)
    return matched_classes, matched_regs, matched_weights


def bbox_encode(bboxes, anchors, scale_factors=None):
    """Bbox encoder (y1, x1, y2, x2) -> (ty, tx, th, tw)
    Args:
      bboxes: Tensor of shape [N, 4]
      anchors: Tensor of shape [M, 4]
      scale_factors: 2D array e.g. [10., 5.]
    Returns:
      encoded bboxes: Tensor of shape [N, 4]
        """
    def _get_center_and_sizes(boxes_):
        yx_min, yx_max = boxes_[:, :2], boxes_[:, 2:]
        hw_ = yx_max - yx_min
        yx_center_ = yx_min + 0.5 * hw_
        return yx_center_, hw_

    # Convert anchors to the center coordinate representation.
    yx_center_a, hw_a = _get_center_and_sizes(anchors)
    yx_center, hw = _get_center_and_sizes(bboxes)
    # Avoid NaN in division and log below.
    hw_a += EPSILON
    hw += EPSILON

    t_yx = (yx_center - yx_center_a) / hw_a
    t_hw = tf.log(hw / hw_a)
    # Scales location targets as used in paper for joint training.
    if scale_factors is not None:
        t_yx *= scale_factors[0]
        t_hw *= scale_factors[1]
    return tf.concat([t_yx, t_hw], axis=-1)


def bbox_decode(bboxes, anchors, scale_factors=None):
    """Bbox decoder (ty, tx, th, tw) -> (y1, x1, y2, x2)
    Args:
      bboxes: Tensor of shape [N, 4]
      anchors: Tensor of shape [N, 4]
      scale_factors: 2D array e.g. [10., 5.]
    Returns:
      decoded bboxes: Tensor of shape [N, 4]
        """
    def _get_center_and_sizes(boxes_):
        yx_min, yx_max = boxes_[:, :2], boxes_[:, 2:]
        hw_ = yx_max - yx_min
        yx_center_ = yx_min + 0.5 * hw_
        return yx_center_, hw_

    # Convert anchors to the center coordinate representation.
    yx_center_a, hw_a = _get_center_and_sizes(anchors)
    t_yx, t_hw = bboxes[:, :2], bboxes[:, 2:]

    if scale_factors is not None:
        t_yx /= scale_factors[0]
        t_hw /= scale_factors[1]

    hw = tf.exp(t_hw) * hw_a
    yx_center = t_yx * hw_a + yx_center_a
    yx_min = yx_center - 0.5 * hw
    yx_max = yx_center + 0.5 * hw

    return tf.concat([yx_min, yx_max], axis=-1)


def generate_anchors(grid_shape, base_anchor_size, stride,
                     scales, aspect_ratios):
    """Generates anchors
    Args:
      grid_shape: [h, w] spatial shape of the feature map over which
        to generate anchors.
      base_anchor_size: Anchor size before applying scaling/aspect ratios
      stride: Stride of anchors on the feature map. For example, if the
        value is 2 then generate anchors for every other feature map pixel.
      scales: 1D array of scales to apply on base anchor. Example: [1., 1.5]
      aspect_ratios: 1D array of anchor ratios (w / h). Example: [0.5, 1., 2.]
    Returns:
      Anchors: Tensor of shape [N, 4]
        where N = grid_h * grid_w * (bboxes per anchor)
        and bboxes have format (y1, x1, y2, x2)
    """
    scales, aspect_ratios = tf.meshgrid(scales, aspect_ratios)
    scales = tf.squeeze(scales)
    ratio_sqrts = tf.sqrt(tf.squeeze(aspect_ratios))
    heights = scales / ratio_sqrts * base_anchor_size
    widths = scales * ratio_sqrts * base_anchor_size

    # Get a grid of box centers
    grid_height, grid_width = grid_shape[0], grid_shape[1]
    heights = heights / tf.to_float(grid_height * stride)
    widths = widths / tf.to_float(grid_width * stride)
    y_centers = 0.5 + tf.to_float(tf.range(grid_height))
    y_centers = y_centers / tf.to_float(grid_height)
    x_centers = 0.5 + tf.to_float(tf.range(grid_width))
    x_centers = x_centers / tf.to_float(grid_width)
    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = tf.meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = tf.meshgrid(heights, y_centers)
    bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=2)
    bbox_sizes = tf.stack([heights_grid, widths_grid], axis=2)
    bbox_centers = tf.reshape(bbox_centers, [-1, 2])
    bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
    bboxes = tf.concat([bbox_centers - 0.5 * bbox_sizes,
                        bbox_centers + 0.5 * bbox_sizes], axis=1)
    return bboxes
