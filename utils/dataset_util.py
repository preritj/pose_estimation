import tensorflow as tf


def normalize_bboxes(bboxes, img_shape):
    img_shape = tf.cast(img_shape, tf.float32)
    img_h, img_w = tf.split(value=img_shape, num_or_size_splits=2)
    ymin, xmin, h, w = tf.split(value=bboxes, num_or_size_splits=4, axis=1)
    return tf.concat([ymin / img_h, xmin / img_w, h / img_h, w / img_w], 1)


def normalize_keypoints(keypoints, img_shape):
    img_shape = tf.cast(img_shape, tf.float32)
    img_h, img_w = tf.split(value=img_shape, num_or_size_splits=2)
    x, y, v = tf.split(value=keypoints, num_or_size_splits=3, axis=1)
    return tf.concat([x / img_w, y / img_h, v], 1)


def random_flip_left_right(img, keypoints, bboxes, mask):
    random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])

    def _flip_keypoints(kps):
        x, y, v = tf.split(value=kps, num_or_size_splits=4, axis=1)
        return tf.concat([1. - x, y, v], 1)

    def _flip_bboxes(bboxes):
        ymin, xmin, h, w = tf.split(value=bboxes, num_or_size_splits=4,
                                    axis=1)
        return tf.concat([ymin, 1. - xmin, h, w], 1)

    flipped_img = tf.cond(random_var,
                          true_fn=lambda: tf.image.flip_left_right(img),
                          false_fn=lambda: tf.identity(img))
    flipped_mask = tf.cond(random_var,
                           true_fn=lambda: tf.image.flip_left_right(mask),
                           false_fn=lambda: tf.identity(mask))
    flipped_keypoints = tf.cond(random_var,
                                true_fn=lambda: _flip_keypoints(keypoints),
                                false_fn=lambda: tf.identity(keypoints))
    flipped_bbox = tf.cond(random_var,
                           true_fn=lambda: _flip_bboxes(bboxes),
                           false_fn=lambda: tf.identity(bboxes))
    return flipped_img, flipped_keypoints, flipped_bbox, flipped_mask