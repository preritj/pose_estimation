import tensorflow as tf
import cv2
import numpy as np
import os
import time


img_h, img_w = (1080, 1920)
net_input_h, net_input_w = (320, 320)
# create batch of image patches of appropriate sizes
# recommended patch dimension is 2 to 4 times bbox dimension
# where dimension is defined as sqrt(h * w)
# aspect ratio must be preserved even if patches overlap
# in fact, overlapping patches should be encouraged!

# The following setting works well for Walmart videos:
patch_h, patch_w = (800, 800)
strides_rows, strides_cols = (280, 560)

# The following setting works well for Recording 44:
# patch_h, patch_w = 1080, 1080
# strides_rows, strides_cols = (1, 840)


n_rows = int(np.ceil(img_h / patch_h))
n_cols = int(np.ceil(img_w / patch_w))
patches_top = np.repeat(strides_rows *np.arange(n_rows), n_cols)
patches_left = np.tile(strides_cols * np.arange(n_cols), n_rows)
patches_top_left = np.array([patches_top, patches_left]).T


def create_patches(image):
    """tensorflow implementation for patch extraction"""
    images = tf.expand_dims(image, axis=0)
    # TODO: resize image first
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
    patches = tf.image.resize_images(
        patches, size=(net_input_h, net_input_w))
    return patches


def create_patches_v2(image):
    """custom implementation of patch extraction:
    [currently not used]
    NOTE: run speed similar to create_patch"""

    # TODO: resize image first
    def _extract_patch(patches_top_left_):
        y, x = patches_top_left_
        return image[y:y + patch_h, x:x + patch_w]

    patches = tf.map_fn(_extract_patch,
                        elems=(patches_top, patches_left),
                        back_prop=False,
                        parallel_iterations=12,  # CPU cores?
                        dtype=tf.float32)
    patches = tf.image.resize_images(
        patches, size=(net_input_h, net_input_w))
    return patches


def stitch_patches(patches):
    pass


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def

    # # Then, we import the graph_def into a new Graph and returns it
    # with tf.Graph().as_default() as graph:
    #     input_image = tf.placeholder(tf.float32, shape=[1080, 1920, 3],
    #                                  name='placeholder/image')
    #     tf_batch_images = tf_batch(input_image)
    #     # The name var will prefix every op/nodes in your graph
    #     # Since we load everything in a new graph, this is not needed
    #     tf.import_graph_def(graph_def)
    # return graph


# preprocess_graph = tf.Graph()
# with preprocess_graph.as_default():
#     input_image = tf.placeholder(tf.float32, shape=[1080, 1920, 3])
#     tf_batch_images = tf_batch(input_image)

frozen_model_filename = 'models/latest/frozen_model.pb'
# graph = load_graph(frozen_model_filename)
# input_image = graph.get_tensor_by_name('placeholder/image:0')
# tf_batch_images = graph.get_tensor_by_name('batch_images')
# tf_images = graph.get_tensor_by_name('import/images:0')
# heatmap = graph.get_tensor_by_name('import/heatmaps:0')

# postprocess_graph = tf.Graph()
# with postprocess_graph.as_default():
#     tf_heatmaps = tf.placeholder(tf.float32, shape=[6, 40, 40, 3])
#     tf_out = tf.zeros((432, 768, 3))
#     def map_fn(in_, heatmap_):
#         out = tf.maximum()
#
#     for i in range(6):
#         heatmap_plane = tf.foldl(map_fn, tf_heatmaps[i],
#                                  initializer=heatmap_plane,
#                                  back_prop=False)


def run_inference(img_files):
    input_image = tf.placeholder(tf.float32, shape=[1080, 1920, 3])
    tf_batch_images = create_patches(input_image)

    graph_def = load_graph(frozen_model_filename)
    with tf.get_default_graph().as_default() as g:
        tf.import_graph_def(graph_def)
    tf_images = g.get_tensor_by_name('import/images:0')
    heatmap = g.get_tensor_by_name('import/heatmaps:0')
    sess = tf.Session(graph=g)
    sum_t1, sum_t2, sum_t3 = 0., 0., 0.

    n_skip = 3
    n_frames = len(img_files)
    for count, img_file in enumerate(img_files):
        # read image 1080 x 1920
        image = cv2.imread(img_file)
        # tensorflow expects RGB!
        image = image[:, :, ::-1]  # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

        out = np.zeros_like(image, dtype=np.float32)
        # combine the patches using some logic
        # e.g. here I simply use maximum
        for i, (y0, x0) in enumerate(patches_top_left):
            out[y0:y0 + patch_h, x0:x0 + patch_h] = np.maximum(
                out[y0:y0 + patch_h, x0:x0 + patch_h],
                cv2.resize(heatmap_pred[i], (patch_w, patch_h)))
        # some post-processing
        threshold = 0.25
        out[out > threshold] = 1.
        out[out < threshold] = 0.
        out = (255. * out).astype(np.uint8)

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


if __name__ == "__main__":
    data_dir = '/media/easystore/TrainData/Walmart/Round1/Recording_2'
    # data_dir = '/media/easystore/TrainData/Lab/April20/Recording_44'
    img_files = []
    for img_id in range(50):
        img_file = '20180308_' + str(img_id).zfill(7) + '.jpg'
        # img_file = '20180420_' + str(img_id).zfill(7) + '.jpg'
        img_files.append(os.path.join(data_dir, img_file))
    run_inference(img_files)
