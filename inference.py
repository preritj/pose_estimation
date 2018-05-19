import tensorflow as tf
import cv2
import numpy as np
import os
import time


net_input_h, net_input_w = 320, 320
# create batch of image patches of appropriate sizes
# recommended patch dimension is 2 to 4 times bbox dimension
# where dimension is defined as sqrt(h * w)
# aspect ratio must be preserved even if patches overlap
# in fact, overlapping patches should be encouraged!

# The following setting works well for Walmart videos:
patches_top_left = [[0, 0], [0, 560], [0, 1120],
                    [280, 0], [280, 560], [280, 1120]]
patches_top = [0, 0, 0, 280, 280, 280]
patches_left = [0, 560, 1120, 0, 560, 1120]
patch_h, patch_w = 800, 800
strides_rows, strides_cols = 280, 560


# The following setting works well for Recording 44:
# patches_top_left = [[0, 0], [0, 840]]
# patch_h, patch_w = 1080, 1080


def tf_batch(image):
    """tensorflow implementation for patch extarction"""
    images = tf.expand_dims(image, axis=0)
    batch_images = tf.extract_image_patches(
        images,
        ksizes=[1, patch_h, patch_w, 1],
        strides=[1, strides_rows, strides_cols, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
        name=None
    )
    batch_images = tf.reshape(
        batch_images, (len(patches_top), patch_h, patch_w, 3))
    batch_images = tf.image.resize_images(
        batch_images, size=(net_input_h, net_input_w))
    return batch_images


def tf_batch_v2(image):
    """my implementation of patch extraction:
    NOTE: this function gives about same speed as tf_batch"""

    def _extract_patch(input_):
        y, x = input_
        return image[y:y + patch_h, x:x + patch_w]

    elems = (np.array(patches_top), np.array(patches_left))
    batch_images = tf.map_fn(_extract_patch,
                             elems,
                             back_prop=False,
                             parallel_iterations=12,  # CPU cores?
                             dtype=tf.float32)
    batch_images = tf.image.resize_images(
        batch_images, size=(net_input_h, net_input_w))
    return batch_images


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
    tf_batch_images = tf_batch(input_image)

    graph_def = load_graph(frozen_model_filename)
    with tf.get_default_graph().as_default() as g:
        tf.import_graph_def(graph_def)
    tf_images = g.get_tensor_by_name('import/images:0')
    heatmap = g.get_tensor_by_name('import/heatmaps:0')
    sess = tf.Session(graph=g)
    t1, t2, t3 = 0., 0., 0.

    n_frames = len(img_files)
    for count, img_file in enumerate(img_files):
        # read image 1080 x 1920
        image = cv2.imread(img_file)
        # tensorflow expects RGB!
        image = image[:, :, ::-1]  # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start = time.time()
        batch_images = sess.run(tf_batch_images, feed_dict={input_image: image})
        if count > 0:  # skip first inference
            t1 += time.time() - start

        # get inference time for forward pass
        heatmap_pred = sess.run(heatmap, feed_dict={tf_images: batch_images})
        if count > 0:  # skip first inference
            t2 += time.time() - start

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
        t1 / (n_frames - 1) * 1000))
    print("Network inference time : {:5.2f} ms/image".format(
        t2 / (n_frames - 1) * 1000))


if __name__ == "__main__":
    data_dir = '/media/easystore/TrainData/Walmart/Round1/Recording_2'
    # data_dir = '/media/easystore/TrainData/Lab/April20/Recording_44'
    img_files = []
    for img_id in range(50):
        img_file = '20180308_' + str(img_id).zfill(7) + '.jpg'
        # img_file = '20180420_' + str(img_id).zfill(7) + '.jpg'
        img_files.append(os.path.join(data_dir, img_file))
    run_inference(img_files)
