import tensorflow as tf


def random_int(maxval, minval=0):
    return tf.random_uniform(
        minval=minval, maxval=maxval, dtype=tf.int32, shape=[])


def normalize_bboxes(bboxes, img_shape):
    img_shape = tf.cast(img_shape, tf.float32)
    img_h, img_w = tf.split(value=img_shape, num_or_size_splits=2)
    ymin, xmin, ymax, xmax = tf.split(value=bboxes, num_or_size_splits=4, axis=1)
    return tf.concat([ymin / img_h, xmin / img_w, ymax / img_h, xmax / img_w], 1)


def normalize_keypoints(keypoints, img_shape):
    img_shape = tf.cast(img_shape, tf.float32)
    img_h, img_w = tf.split(value=img_shape, num_or_size_splits=2)
    x, y, v = tf.split(value=keypoints, num_or_size_splits=3, axis=2)
    v = tf.minimum(v, 1)
    return tf.concat([x / img_w, y / img_h, v], 2)


def flip_left_right_keypoints(keypoints):
    x, y, v = tf.split(value=keypoints, num_or_size_splits=3, axis=2)
    return tf.concat([1. - x, y, v], 2)


def flip_left_right_bboxes(bboxes):
    ymin, xmin, ymax, xmax = tf.split(value=bboxes, num_or_size_splits=4,
                                      axis=1)
    return tf.concat([ymin, 1. - xmin, ymax, 1. - xmax], 1)


def random_flip_left_right(img, keypoints, bboxes, mask):
    random_var = random_int(2)
    random_var = tf.cast(random_var, tf.bool)
    flipped_img = tf.cond(random_var,
                          true_fn=lambda: tf.image.flip_left_right(img),
                          false_fn=lambda: tf.identity(img))
    mask = tf.expand_dims(mask, axis=2)
    flipped_mask = tf.cond(random_var,
                           true_fn=lambda: tf.image.flip_left_right(mask),
                           false_fn=lambda: tf.identity(mask))
    flipped_mask = tf.squeeze(flipped_mask)
    flipped_keypoints = tf.cond(
        random_var,
        true_fn=lambda: flip_left_right_keypoints(keypoints),
        false_fn=lambda: tf.identity(keypoints))
    flipped_bbox = tf.cond(
        random_var,
        true_fn=lambda: flip_left_right_bboxes(bboxes),
        false_fn=lambda: tf.identity(bboxes))
    return flipped_img, flipped_keypoints, flipped_bbox, flipped_mask


def prune_bboxes_keypoints(bboxes, keypoints, crop_box):
    ymin, xmin, ymax, xmax = tf.split(value=bboxes, num_or_size_splits=4,
                                      axis=1)
    crop_ymin, crop_xmin, crop_ymax, crop_xmax = tf.unstack(crop_box)
    crop_h, crop_w = crop_ymax - crop_ymin, crop_xmax - crop_xmin
    ymin, xmin = (ymin - crop_ymin) / crop_h, (xmin - crop_xmin) / crop_w
    ymax, xmax = (ymax - crop_ymin) / crop_h, (xmax - crop_xmin) / crop_w
    is_outside = tf.concat([
        tf.greater_equal(ymin, 1.), tf.greater_equal(xmin, 1.),
        tf.less_equal(ymax, 0.), tf.less_equal(xmax, 0.)
    ], 1)
    is_outside = tf.reduce_any(is_outside, 1)
    valid_bboxes = tf.concat([ymin, xmin, ymax, xmax], axis=1)
    valid_bboxes = tf.boolean_mask(valid_bboxes, tf.logical_not(is_outside))
    valid_bboxes = tf.clip_by_value(valid_bboxes,
                                    clip_value_min=0.,
                                    clip_value_max=1.)
    valid_keypoints = tf.boolean_mask(keypoints, tf.logical_not(is_outside))
    x, y, v = tf.split(value=valid_keypoints, num_or_size_splits=3, axis=2)
    x, y = (x - crop_xmin) / crop_w, (y - crop_ymin) / crop_h
    is_outside = tf.concat([
        tf.greater_equal(x, 1.), tf.greater_equal(y, 1.),
        tf.less_equal(x, 0.), tf.less_equal(y, 0.)
    ], 2)
    is_outside = tf.reduce_any(is_outside, 2)
    is_outside = tf.cast(tf.logical_not(is_outside), tf.float32)
    v = v * tf.expand_dims(is_outside, 2)
    valid_keypoints = tf.concat([x, y, v], axis=2)
    return valid_bboxes, valid_keypoints


def random_crop(image, keypoints, bboxes, mask, crop_size=None):
    # treat each bbox as a batch, so reshape from [N, 4] to [N, 1, 4]
    boxes_expanded = tf.expand_dims(tf.clip_by_value(
        bboxes, clip_value_min=0.0, clip_value_max=1.0), 1)
    # require patch to have one full bbox
    img_begin, img_size, crop_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        boxes_expanded,
        min_object_covered=1.,
        area_range=[0.1, 1.],
        aspect_ratio_range=[.75, 1.33],
        use_image_if_no_bounding_boxes=True
    )
    crop_box = tf.squeeze(crop_box)
    new_image = tf.slice(image, img_begin, img_size)
    new_image.set_shape([None, None, image.get_shape()[2]])
    mask_begin = img_begin[:2]
    mask_size = img_size[:2]
    new_mask = tf.slice(mask, mask_begin, mask_size)
    new_bboxes, new_keypoints = prune_bboxes_keypoints(
        bboxes, keypoints, crop_box)
    if crop_size is None:
        return new_image, new_keypoints, new_bboxes, new_mask

    img_shape = tf.cast(tf.shape(new_image), tf.float32)
    img_h, img_w = img_shape[0], img_shape[1]
    img_aspect_ratio = img_w / img_h
    crop_aspect_ratio = tf.constant(1. * crop_size[1] / crop_size[0],
                                    dtype=tf.float32)

    def target_height_fn():
        return tf.to_int32(tf.round(img_w / crop_aspect_ratio))

    crop_h = tf.cond(img_aspect_ratio >= crop_aspect_ratio,
                     true_fn=lambda: tf.to_int32(img_h),
                     false_fn=target_height_fn)

    def target_width_fn():
        return tf.to_int32(tf.round(img_h * crop_aspect_ratio))

    crop_w = tf.cond(img_aspect_ratio <= crop_aspect_ratio,
                     true_fn=target_width_fn,
                     false_fn=lambda: tf.to_int32(img_w))

    bbox_min, bbox_max = tf.split(new_bboxes, num_or_size_splits=[2, 2],
                                  axis=1)
    bbox_min = tf.to_int32(tf.round(bbox_min * img_shape[:2]))
    bbox_max = tf.to_int32(tf.round(bbox_max * img_shape[:2]))
    bbox_min = tf.reduce_min(bbox_min, 0)
    bbox_max = tf.reduce_max(bbox_max, 0)
    print(bbox_min)
    crop_shape = tf.stack([crop_h, crop_w])
    offset_min = tf.maximum(0, bbox_max - crop_shape)
    offset_max = tf.minimum(tf.to_int32(img_shape[:2]) - crop_shape + 1,
                            bbox_min + 1)
    minval = tf.minimum(offset_min, offset_max)
    maxval = tf.maximum(offset_min, offset_max)

    offset_h = random_int(maxval=maxval[0], minval=minval[0])
    offset_w = random_int(maxval=maxval[1], minval=minval[1])
    new_image = tf.image.crop_to_bounding_box(
        new_image, offset_h, offset_w, crop_h, crop_w)
    new_mask = tf.expand_dims(new_mask, 2)
    new_mask = tf.image.crop_to_bounding_box(
        new_mask, offset_h, offset_w, crop_h, crop_w)
    new_mask = tf.squeeze(new_mask)
    crop_box = tf.stack([
        tf.to_float(offset_h) / img_h,
        tf.to_float(offset_w) / img_w,
        tf.to_float(offset_h + crop_h) / tf.to_float(img_h),
        tf.to_float(offset_w + crop_w) / tf.to_float(img_w)
    ])
    new_bboxes, new_keypoints = prune_bboxes_keypoints(
        new_bboxes, new_keypoints, crop_box)
    return new_image, new_keypoints, new_bboxes, new_mask
