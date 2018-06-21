import tensorflow as tf
import numpy as np
import cv2


def random_int(maxval, minval=0):
    return tf.random_uniform(
        shape=[], minval=minval, maxval=maxval, dtype=tf.int32)


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


def keypoints_select(img, keypoints, bboxes, mask, keypoints_to_keep):
    keypoints_subset = tf.gather(keypoints,
                                 keypoints_to_keep,
                                 axis=1)
    return img, keypoints_subset, bboxes, mask


def flip_left_right_keypoints(keypoints, flipped_keypoint_indices):
    x, y, v = tf.split(value=keypoints, num_or_size_splits=3, axis=2)
    flipped_keypoints = tf.concat([1. - x, y, v], 2)
    flipped_keypoints = tf.gather(flipped_keypoints,
                                  flipped_keypoint_indices,
                                  axis=1)
    return flipped_keypoints


def flip_left_right_bboxes(bboxes):
    ymin, xmin, ymax, xmax = tf.split(value=bboxes, num_or_size_splits=4,
                                      axis=1)
    return tf.concat([ymin, 1. - xmax, ymax, 1. - xmin], 1)


def random_flip_left_right(img, keypoints, bboxes, mask,
                           flipped_keypoint_indices):
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
        true_fn=lambda: flip_left_right_keypoints(
            keypoints, flipped_keypoint_indices),
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
        tf.greater(ymin, 1.), tf.greater(xmin, 1.),
        tf.less(ymax, 0.), tf.less(xmax, 0.)
    ], 1)
    is_outside = tf.reduce_any(is_outside, 1)
    valid_indices = tf.reshape(tf.where(tf.logical_not(is_outside)), [-1])
    valid_bboxes = tf.concat([ymin, xmin, ymax, xmax], axis=1)
    valid_bboxes = tf.gather(valid_bboxes, valid_indices)
    valid_bboxes = tf.clip_by_value(valid_bboxes,
                                    clip_value_min=0.,
                                    clip_value_max=1.)
    valid_keypoints = tf.gather(keypoints, valid_indices)
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


def random_crop(image, keypoints, bboxes, mask,
                crop_size=(224, 224), scale_range=(1.5, 4.)):
    bboxes = tf.clip_by_value(
        bboxes, clip_value_min=0.0, clip_value_max=1.0)
    n_bboxes = tf.shape(bboxes)[0]
    img_shape = tf.cast(tf.shape(image), tf.float32)
    img_h, img_w = img_shape[0], img_shape[1]
    random_bbox = tf.cond(
        tf.greater(n_bboxes, 0),
        true_fn=lambda: tf.random_shuffle(bboxes)[0],
        false_fn=lambda: tf.constant([0., 0., 1., 1.]))
    bbox_area = ((random_bbox[2] - random_bbox[0])
                 * (random_bbox[3] - random_bbox[1]))
    img_aspect_ratio = img_w / img_h
    aspect_ratio = 1. * crop_size[1] / crop_size[0]
    crop_aspect_ratio = tf.constant(aspect_ratio, tf.float32)

    def target_height_fn():
        return tf.to_int32(tf.round(img_w / crop_aspect_ratio))

    crop_h = tf.cond(img_aspect_ratio >= aspect_ratio,
                     true_fn=lambda: tf.to_int32(img_h),
                     false_fn=target_height_fn)

    def target_width_fn():
        return tf.to_int32(tf.round(img_h * crop_aspect_ratio))

    crop_w = tf.cond(img_aspect_ratio <= aspect_ratio,
                     true_fn=lambda: tf.to_int32(img_w),
                     false_fn=target_width_fn)

    max_crop_shape = tf.stack([crop_h, crop_w])

    crop_area = 1. * crop_size[0] * crop_size[1]
    scale_ratio = tf.sqrt(bbox_area * img_h * img_w / crop_area)

    crop_size = tf.constant(list(crop_size), tf.float32)
    # cap min scale at 0.5 to prevent bad resolution
    scale_min = tf.maximum(scale_range[0] * scale_ratio, 0.5)
    # max scale has to be greater than min scale (1.1 * min scale)
    scale_max = tf.maximum(scale_range[1] * scale_ratio,
                           1.1 * scale_min)
    size_min = tf.minimum(max_crop_shape - 1,
                          tf.to_int32(scale_min * crop_size))
    size_max = tf.minimum(max_crop_shape,
                          tf.to_int32(scale_max * crop_size))
    crop_h = random_int(maxval=tf.to_int32(size_max[0]),
                        minval=tf.to_int32(size_min[0]))
    crop_w = tf.to_int32(aspect_ratio * tf.to_float(crop_h))
    crop_shape = tf.stack([crop_h, crop_w])

    bbox_min, bbox_max = random_bbox[:2], random_bbox[2:]
    bbox_min = tf.cast(tf.round(bbox_min * img_shape[:2]), tf.int32)
    bbox_max = tf.cast(tf.round(bbox_max * img_shape[:2]), tf.int32)
    bbox_min = tf.maximum(bbox_min, 0)

    offset_min = tf.maximum(0, bbox_max - crop_shape)
    offset_max = tf.minimum(
        tf.cast(img_shape[:2], tf.int32) - crop_shape + 1,
        bbox_min + 1)
    offset_min = tf.where(tf.less_equal(offset_max, offset_min),
                          tf.constant([0, 0]),
                          offset_min)

    offset_h = random_int(maxval=offset_max[0], minval=offset_min[0])
    offset_w = random_int(maxval=offset_max[1], minval=offset_min[1])

    new_image = tf.image.crop_to_bounding_box(
        image, offset_h, offset_w, crop_h, crop_w)
    new_mask = tf.expand_dims(mask, 2)
    new_mask = tf.image.crop_to_bounding_box(
        new_mask, offset_h, offset_w, crop_h, crop_w)
    new_mask = tf.squeeze(new_mask)
    crop_box = tf.stack([
        tf.to_float(offset_h) / img_h,
        tf.to_float(offset_w) / img_w,
        tf.to_float(offset_h + crop_h) / img_h,
        tf.to_float(offset_w + crop_w) / img_w
    ])
    new_bboxes, new_keypoints = prune_bboxes_keypoints(
        bboxes, keypoints, crop_box)
    return new_image, new_keypoints, new_bboxes, new_mask


def resize(image, keypoints, bbox, mask,
           target_image_size=(224, 224),
           target_mask_size=None):
    img_size = list(target_image_size)
    if target_mask_size is None:
        target_mask_size = img_size
    mask_size = list(target_mask_size)
    new_image = tf.image.resize_images(image, size=img_size)
    new_mask = tf.expand_dims(mask, axis=2)
    new_mask.set_shape([None, None, 1])
    new_mask = tf.image.resize_images(new_mask, size=mask_size)
    new_mask = tf.squeeze(new_mask)
    return new_image, keypoints, bbox, new_mask


def _generate_heatmap_plane(center, sigma, shape):
    roi_min = tf.cast(tf.maximum(center - 2 * sigma, 0), tf.int32)
    roi_max = tf.cast(
        tf.minimum(center + 2 * sigma, tf.to_float(shape)),
        tf.int32)
    x = tf.range(roi_min[0], roi_max[0], dtype=tf.int32)
    y = tf.range(roi_min[1], roi_max[1], dtype=tf.int32)
    x, y = tf.meshgrid(x, y)
    d = tf.square(tf.to_float(x) - center[0]) \
        + tf.square(tf.to_float(y) - center[1])
    intensity = tf.exp(- d / sigma / sigma)
    indices = tf.stack([tf.reshape(y, [-1]), tf.reshape(x, [-1])], 1)
    values = tf.reshape(intensity, [-1])
    heatmap = tf.SparseTensor(tf.to_int64(indices), values,
                              dense_shape=shape)
    return tf.cast(tf.sparse_tensor_to_dense(heatmap), tf.float32)


def _get_heatmap(keypoints, sigma, shape):
    keypoints_ = tf.transpose(keypoints, [1, 0, 2])
    heatmaps = []

    def map_fn(heatmap_plane_, keypoint):
        center, is_visible = keypoint[:2], keypoint[2]
        center *= tf.to_float(tf.stack([shape[1], shape[0]]))
        return tf.cond(
            tf.greater(is_visible, 0),
            true_fn=lambda: tf.maximum(
                heatmap_plane_,
                _generate_heatmap_plane(center, sigma, shape)),
            false_fn=lambda: heatmap_plane_)

    for kps in tf.unstack(keypoints_):
        heatmap_plane = tf.zeros(shape, dtype=tf.float32)
        heatmap_plane = tf.foldl(map_fn, kps,
                                 initializer=heatmap_plane,
                                 back_prop=False)
        heatmaps.append(heatmap_plane)
    return tf.stack(heatmaps)


def keypoints_to_heatmap_v1(image, keypoints, bboxes, mask,
                            num_keypoints=15, sigma=8.):
    """CMU version of generating heatmaps"""
    shape = tf.cast(tf.shape(image)[:2], tf.int64)
    keypoints.set_shape([None, num_keypoints, 3])
    heatmap = _get_heatmap(keypoints, sigma, shape)
    return image, heatmap, bboxes, mask


def keypoints_to_heatmap(image, keypoints, bboxes, mask,
                         num_keypoints=15,
                         grid_shape=(28, 28)):
    """Mask-rcnn method of generating heatmaps"""
    keypoints = tf.clip_by_value(keypoints,
                                 clip_value_min=0.,
                                 clip_value_max=.999)
    grid_shape_ = tf.constant(list(grid_shape), tf.int32)
    keypoints.set_shape([None, num_keypoints, 3])
    num_instances = tf.shape(keypoints)[0]
    kp_indices = tf.range(0, num_keypoints, dtype=tf.int32)
    kp_indices = tf.tile(kp_indices, [num_instances])
    kp_indices = tf.reshape(kp_indices, [-1, 1])
    n_indices = tf.range(0, num_instances, dtype=tf.int32)
    n_indices = tf.tile(
        tf.reshape(n_indices, [-1, 1]), [1, num_keypoints])
    n_indices = tf.reshape(n_indices, [-1, 1])
    keypoints_ = tf.reshape(keypoints, [-1, 3])
    keypoints_x, keypoints_y, keypoints_vis = tf.split(
        keypoints_, num_or_size_splits=3, axis=1)
    x_indices = tf.to_int32(tf.floor(
        keypoints_x * tf.to_float(grid_shape_[1])))
    y_indices = tf.to_int32(tf.floor(
        keypoints_y * tf.to_float(grid_shape_[0])))
    indices = tf.to_int64(tf.concat(
        [n_indices, kp_indices, y_indices, x_indices],
        axis=1))
    keypoints_vis = tf.squeeze(keypoints_vis, [1])
    values = tf.cast(tf.greater(keypoints_vis, 0.), tf.int32)
    dense_shape = tf.to_int64(tf.concat(
        [tf.shape(keypoints)[:2], grid_shape_], axis=0))
    heatmap = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=dense_shape)
    shape = [-1, num_keypoints, grid_shape[0], grid_shape[1]]
    heatmap = tf.sparse_reshape(heatmap, shape=shape)
    heatmap = tf.sparse_reduce_max_sparse(heatmap, axis=0)
    heatmap = tf.sparse_transpose(heatmap, [1, 2, 0])
    heatmap = tf.sparse_tensor_to_dense(heatmap)
    return image, heatmap, bboxes, mask


def keypoints_to_heatmap_v2(image, keypoints, bboxes, mask,
                           num_keypoints=15,
                           grid_shape=(28, 28),
                           window_size=3):
    """Heatmap generation"""
    n = window_size * window_size
    keypoints = tf.clip_by_value(keypoints,
                                 clip_value_min=0.,
                                 clip_value_max=.999)
    grid_shape_ = tf.constant(list(grid_shape), tf.int32)
    keypoints.set_shape([None, num_keypoints, 3])
    num_instances = tf.shape(keypoints)[0]
    kp_indices = tf.range(0, num_keypoints, dtype=tf.int32)
    kp_indices = tf.tile(
        tf.expand_dims(kp_indices, axis=1),
        [num_instances, n])
    kp_indices = tf.reshape(kp_indices, [-1, 1])
    n_indices = tf.range(0, num_instances, dtype=tf.int32)
    n_indices = tf.tile(
        tf.reshape(n_indices, [-1, 1]),
        [1, n * num_keypoints])
    n_indices = tf.reshape(n_indices, [-1, 1])
    keypoints_ = tf.reshape(keypoints, [-1, 3])
    keypoints_x, keypoints_y, keypoints_vis = tf.split(
        keypoints_, num_or_size_splits=3, axis=1)

    center_x_indices = tf.to_int32(tf.floor(
        keypoints_x * tf.to_float(grid_shape_[1])))
    center_y_indices = tf.to_int32(tf.floor(
        keypoints_y * tf.to_float(grid_shape_[0])))
    delta = int((window_size - 1) / 2)
    x_indices, y_indices = [], []
    for i in range(-delta, delta + 1):
        for j in range(-delta, delta + 1):
            new_x_indices = tf.clip_by_value(
                center_x_indices + i,
                clip_value_min=0,
                clip_value_max=grid_shape[1] - 1)
            new_y_indices = tf.clip_by_value(
                center_y_indices + j,
                clip_value_min=0,
                clip_value_max=grid_shape[0] - 1)
            x_indices.append(new_x_indices)
            y_indices.append(new_y_indices)
    x_indices = tf.reshape(tf.transpose(
        tf.concat(x_indices, axis=1), [0, 1]), [-1, 1])
    y_indices = tf.reshape(tf.transpose(
        tf.concat(y_indices, axis=1), [0, 1]), [-1, 1])

    indices = tf.to_int64(tf.concat(
        [n_indices, kp_indices, y_indices, x_indices],
        axis=1))

    keypoints_vis = tf.squeeze(keypoints_vis, [1])
    keypoints_vis = tf.reshape(tf.tile(
        tf.reshape(keypoints_vis, [-1, 1]),
        [1, n]), [-1])
    values = tf.cast(tf.greater(keypoints_vis, 0.), tf.int32)
    dense_shape = tf.to_int64(tf.concat(
        [tf.shape(keypoints)[:2], grid_shape_], axis=0))
    heatmap = tf.SparseTensor(
        indices=indices,
        values=values,
        dense_shape=dense_shape)
    shape = [-1, num_keypoints, grid_shape[0], grid_shape[1]]
    heatmap = tf.sparse_reshape(heatmap, shape=shape)
    heatmap = tf.sparse_reduce_max_sparse(heatmap, axis=0)
    heatmap = tf.sparse_transpose(heatmap, [1, 2, 0])
    heatmap = tf.sparse_tensor_to_dense(heatmap)
    return image, heatmap, bboxes, mask


def keypoints_to_heatmaps_and_vectors(
        keypoints, mask,
        pairs=([0, 1], [1, 2]),
        grid_shape=(28, 28),
        window_size=3,
        vector_scale=20.,
        offset_scale=20):
    num_instances, num_keypoints, _ = keypoints.shape
    n_vec = 4 * len(pairs)
    mask = cv2.resize(mask, (grid_shape[1], grid_shape[0]))
    mask = np.expand_dims(mask, 2)
    grid_shape = np.array(list(grid_shape))
    n = window_size * window_size
    delta = int((window_size - 1) / 2)
    keypoints_vis = keypoints[:, :, 2]
    keypoints_vis = (keypoints_vis > 0).astype(np.uint8)
    keypoints_xy = keypoints[:, :, :2] * grid_shape
    keypoints_indices = (keypoints[:, :, :2] * grid_shape).astype(np.uint16)
    keypoints_xy_delta = keypoints_xy - keypoints_indices - 0.5
    x_center_indices, y_center_indices = keypoints_indices.transpose((2, 0, 1))
    x_center_float, y_center_float = keypoints_xy.transpose((2, 0, 1))
    x_mesh, y_mesh = np.meshgrid(np.arange(-delta, 1 + delta),
                                 np.arange(-delta, 1 + delta))
    bilinear_interp = ((1 - np.absolute(2. * x_mesh) / window_size)
                       * (1 - np.absolute(2. * y_mesh) / window_size))
    bilinear_interp = bilinear_interp.flatten()
    x_mesh, y_mesh = x_mesh.reshape((-1, 1)), y_mesh.reshape((-1, 1))
    heatmap = np.zeros((grid_shape[0], grid_shape[1], num_keypoints))
    heatmap_mask = mask * np.ones((grid_shape[0], grid_shape[1], num_keypoints))
    offsetmap = np.zeros((grid_shape[0], grid_shape[1], num_keypoints, 2))
    offset_distmap = np.ones((grid_shape[0], grid_shape[1], num_keypoints)) * 100
    vecmap = np.zeros((grid_shape[0], grid_shape[1], n_vec))
    for i in range(num_instances):
        instance_mask = np.ones((grid_shape[0], grid_shape[1], num_keypoints))
        instance_heatmap = np.zeros((grid_shape[0], grid_shape[1], num_keypoints))
        instance_offsetmap = 100 * np.ones(
            (grid_shape[0], grid_shape[1], num_keypoints, 2))
        instance_offset_distmap = 100 * np.ones(
            (grid_shape[0], grid_shape[1], num_keypoints))
        instance_vecmap = np.zeros((grid_shape[0], grid_shape[1], n_vec))
        kp_vis = keypoints_vis[i]
        values = kp_vis * np.expand_dims(bilinear_interp, 1)
        values = values.flatten()
        # values = np.tile(kp_vis, int((n - 1) / 2)) * 0.5
        # values = np.concatenate([values, kp_vis, values])
        mask_values = np.tile(kp_vis, n)
        kp_indices = np.repeat(np.expand_dims(np.arange(num_keypoints), 0),
                               n, axis=0)
        vec_indices = np.repeat(np.expand_dims(np.arange(n_vec), 0), n, axis=0)
        x_indices = (x_center_indices[i] + x_mesh)
        y_indices = (y_center_indices[i] + y_mesh)
        x_idx, y_idx = x_indices.flatten(), y_indices.flatten()
        valid_idx = np.where((x_idx >= 0) & (y_idx >= 0)
                             & (x_idx <= grid_shape[1] - 1)
                             & (y_idx <= grid_shape[0] - 1))[0]
        offset_vals = np.concatenate([x_mesh, y_mesh], axis=1)
        offset_delta = np.tile(keypoints_xy_delta[i],
                               (offset_vals.shape[0], 1))
        offset_vals = np.repeat(offset_vals, num_keypoints, axis=0)
        offset_vals = offset_vals + offset_delta
        distances = np.sqrt(offset_vals[:, 0] ** 2 + offset_vals[:, 1] ** 2)
        indices = (y_idx[valid_idx],
                   x_idx[valid_idx],
                   kp_indices.flatten()[valid_idx])
        instance_heatmap[indices] = values[valid_idx]
        instance_mask[indices] = mask_values[valid_idx]
        instance_offsetmap[indices] = offset_vals[valid_idx]
        instance_offset_distmap[indices] = distances[valid_idx]
        heatmap = np.maximum(heatmap, instance_heatmap)
        heatmap_mask = np.minimum(heatmap_mask, instance_mask)
        overwrite = instance_offset_distmap < offset_distmap
        offsetmap[overwrite] = instance_offsetmap[overwrite]

        select_x_indices, select_y_indices, values = [], [], []
        for kp1, kp2 in pairs:
            select_x_indices += 2 * [x_indices[:, kp1]] + 2 * [x_indices[:, kp2]]
            select_y_indices += 2 * [y_indices[:, kp1]] + 2 * [y_indices[:, kp2]]
            if (kp_vis[kp1] == 0) or (kp_vis[kp2] == 0):
                values += 4 * [np.zeros_like(x_indices[:, kp1])]
            else:
                values += [
                    x_center_float[i, kp2] - x_indices[:, kp1] - 0.5,
                    y_center_float[i, kp2] - y_indices[:, kp1] - 0.5,
                    x_center_float[i, kp1] - x_indices[:, kp2] - 0.5,
                    y_center_float[i, kp1] - y_indices[:, kp2] - 0.5]
                # values += [x_center_indices[i, kp2] - x_indices[:, kp1],
                #            y_center_indices[i, kp2] - y_indices[:, kp1],
                #            x_center_indices[i, kp1] - x_indices[:, kp2],
                #            y_center_indices[i, kp1] - y_indices[:, kp2]]
        select_x_indices = np.array(select_x_indices).T.flatten()
        select_y_indices = np.array(select_y_indices).T.flatten()
        valid_idx = np.where((select_x_indices >= 0) & (select_y_indices >= 0)
                             & (select_x_indices <= grid_shape[1] - 1)
                             & (select_y_indices <= grid_shape[0] - 1))[0]
        instance_vecmap[
            select_y_indices[valid_idx],
            select_x_indices[valid_idx],
            vec_indices.flatten()[valid_idx]] = \
            np.array(values).T.flatten()[valid_idx] / vector_scale
        overwrite = (np.absolute(vecmap) < .00001)
        vecmap[overwrite] = instance_vecmap[overwrite]
    offsetmap = np.reshape(offsetmap, (grid_shape[0], grid_shape[1], -1)) / offset_scale
    return (np.float32(heatmap), np.float32(vecmap),
            np.float32(offsetmap), np.float32(heatmap_mask))
