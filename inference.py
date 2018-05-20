import tensorflow as tf
import cv2
import numpy as np
import os
import time
from utils import ops


img_h0, img_w0 = (1080, 1920)
patch_h, patch_w = (320, 320)
out_stride = 8
# frozen_model_filename = 'models/latest/optimized_model.pb'
frozen_model_filename = 'models/latest/frozen_model.pb'

# create image patches of appropriate sizes
# recommended patch dimension is 2 to 4 times bbox dimension
# where dimension is defined as sqrt(h * w)
# aspect ratio must be preserved even if patches overlap
# in fact, overlapping patches are encouraged!

# settings for Walmart videos:
img_h, img_w = (456, 800)
strides_rows, strides_cols = (136, 240)

# settings for Recording 44:
# img_h, img_w = (320, 569)
# strides_rows, strides_cols = (1, 249)


n_rows = int(np.ceil(img_h / patch_h))
n_cols = int(np.ceil(img_w / patch_w))
patches_top = np.repeat(strides_rows * np.arange(n_rows), n_cols)
patches_left = np.tile(strides_cols * np.arange(n_cols), n_rows)
patches_top_left = np.array([patches_top, patches_left]).T


def read_and_resize_image(img_file):
    image = cv2.imread(img_file)
    # tensorflow expects RGB!
    image = image[:, :, ::-1]
    # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_w, img_h))
    return image


def create_patches(image):
    """tensorflow implementation for patch extraction"""
    images = tf.expand_dims(image, axis=0)
    patches = tf.extract_image_patches(
        images,
        ksizes=[1, patch_h, patch_w, 1],
        strides=[1, strides_rows, strides_cols, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
        name=None
    )
    patches = tf.reshape(
        patches, (len(patches_top), patch_h, patch_w, 3))
    return patches


def create_patches_v2(image):
    """custom implementation of patch extraction:
    [currently not used]
    NOTE: run speed similar to create_patch"""

    def _extract_patch(patches_top_left_):
        y, x = patches_top_left_
        return image[y:y + patch_h, x:x + patch_w]

    patches = tf.map_fn(_extract_patch,
                        elems=(patches_top, patches_left),
                        back_prop=False,
                        parallel_iterations=12,  # CPU cores?
                        dtype=tf.float32)
    return patches


def stitch_to_right(patch_left, patch_right):
    overlap_cols = int((patch_w - strides_cols) / out_stride)
    left, left_overlap = tf.split(
        patch_left, [-1, overlap_cols], axis=1)
    right_overlap, right = tf.split(
        patch_right, [overlap_cols, -1], axis=1)
    weights = tf.reshape(
        tf.range(overlap_cols, dtype=tf.float32),
        shape=(1, overlap_cols, 1))
    overlap = (left_overlap * (overlap_cols - 1. - weights)
               + right_overlap * weights) / overlap_cols
    out = tf.concat([left, overlap, right], axis=1)
    return out


def stitch_to_bottom(patch_top, patch_bottom):
    overlap_rows = int((patch_h - strides_rows) / out_stride)
    top, top_overlap = tf.split(
        patch_top, [-1, overlap_rows], axis=0)
    bottom_overlap, bottom = tf.split(
        patch_bottom, [overlap_rows, -1], axis=0)
    weights = tf.reshape(
        tf.range(overlap_rows, dtype=tf.float32),
        shape=(overlap_rows, 1, 1))
    overlap = (top_overlap * (overlap_rows - 1. - weights)
               + bottom_overlap * weights) / overlap_rows
    out = tf.concat([top, overlap, bottom], axis=0)
    return out


def stitch_patches(patches):
    rows = tf.split(patches, num_or_size_splits=n_rows, axis=0)
    stitched_rows = []
    for i, row in enumerate(rows):
        stitched_row = patches[i * n_cols]
        for j in range(1, n_cols):
            stitched_row = stitch_to_right(
                stitched_row, patches[i * n_cols + j])
        stitched_rows.append(stitched_row)

    out = stitched_rows[0]
    for i in range(1, n_rows):
        out = stitch_to_bottom(out, stitched_rows[i])
    out = tf.expand_dims(out, axis=0)
    out = ops.non_max_suppression(out, window_size=3)
    out = tf.squeeze(out)
    return out


def load_graph_def(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def


def run_inference(img_files):
    # pre-preprocessing graph
    input_image = tf.placeholder(tf.float32, shape=[img_h, img_w, 3])
    tf_batch_images = create_patches(input_image)
    # network inference graph
    graph_def = load_graph_def(frozen_model_filename)
    with tf.get_default_graph().as_default() as g:
        tf.import_graph_def(graph_def)
    tf_images = g.get_tensor_by_name('import/images:0')
    heatmap = g.get_tensor_by_name('import/heatmaps:0')
    # post-processing graph
    tf_patches = tf.placeholder(
        tf.float32,
        shape=[n_rows * n_cols, patch_h / out_stride,
               patch_w / out_stride, 3])
    tf_out = stitch_patches(tf_patches)
    sess = tf.Session(graph=g)
    sum_t1, sum_t2, sum_t3 = 0., 0., 0.

    n_skip = 3
    n_frames = len(img_files)
    for count, img_file in enumerate(img_files):
        image = read_and_resize_image(img_file)
        t0 = time.time()
        batch_images = sess.run(tf_batch_images, feed_dict={input_image: image})
        t1 = time.time()
        if count > n_skip:  # skip first few inferences
            sum_t1 += t1 - t0

        # get inference time for forward pass
        heatmap_pred = sess.run(heatmap, feed_dict={tf_images: batch_images})
        t2 = time.time()
        if count > n_skip:  # skip first few inferences
            sum_t2 += t2 - t1

        out = sess.run(tf_out, feed_dict={tf_patches: heatmap_pred})
        # some additional post-processing
        threshold = 0.5
        out[out > threshold] = 1.
        out[out < threshold] = 0.
        out = (255. * out).astype(np.uint8)
        t3 = time.time()
        out = cv2.resize(out, (img_w, img_h))
        if count > n_skip:  # skip first few inferences
            sum_t3 += t3 - t2

        # back to BGR for opencv display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        out = cv2.addWeighted(image, 0.4, out, 0.6, 0)
        cv2.imshow('out', out)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("Pre-processing time : {:5.2f} ms/image".format(
        sum_t1 / (n_frames - n_skip) * 1000))
    print("Network inference time : {:5.2f} ms/image".format(
        sum_t2 / (n_frames - n_skip) * 1000))
    print("Post-processing time : {:5.2f} ms/image".format(
        sum_t3 / (n_frames - n_skip) * 1000))


if __name__ == "__main__":
    data_dir = '/media/easystore/TrainData/Walmart/Round1/Recording_2'
    # data_dir = '/media/easystore/TrainData/Lab/April20/Recording_44'
    img_files = []
    for img_id in range(500):
        img_file = '20180308_' + str(img_id).zfill(7) + '.jpg'
        # img_file = '20180420_' + str(img_id).zfill(7) + '.jpg'
        img_files.append(os.path.join(data_dir, img_file))
    run_inference(img_files)
