import tensorflow as tf
import cv2
import numpy as np
from utils.ops import non_max_suppression


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def)
    return graph


frozen_model_filename = 'models/latest/frozen_model.pb'
graph = load_graph(frozen_model_filename)
for n in graph.as_graph_def().node:
    print(n.name)

# read image 1080 x 1920
image = cv2.imread('/media/easystore/TrainData/Walmart/Round1/Recording_2/20180308_0000150.jpg')
# tensorflow expects RGB!
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# create batch of image patches of appropriate sizes
# recommeded patch dimension is 2 to 4 times bbox dimension
# where dimension is defined as sqrt(h * w)
# aspect ratio must be preserved even if patches overlap
# in fact, overlapping patches should be encouraged!
# The following setting works well for Walmart videos only:
patches_top_left = [[0, 0], [0, 560], [0, 1120],
                    [280, 0], [280, 560], [280, 1120]]
patch_h, patch_w = 800, 800
batch_images = []
for y0, x0 in patches_top_left:
    patch = image[y0:y0 + patch_h, x0:x0 + patch_w]
    patch = cv2.resize(patch, (320, 320))
    batch_images.append(patch)

tf_images = graph.get_tensor_by_name('import/images:0')
heatmap = graph.get_tensor_by_name('import/heatmaps:0')
# heatmaps = non_max_suppression(heatmap, 3)

with tf.Session(graph=graph) as sess:
    heatmap_pred = sess.run(heatmap, feed_dict={tf_images: batch_images})

    stride = 8  # network generates heatmap output of 40 x 40
    out = np.zeros_like(image, dtype=np.float32)
    # combine the patches using some logic
    # e.g. here I simply use maximum
    for i, (y0, x0) in enumerate(patches_top_left):
        out[y0:y0 + patch_h, x0:x0 + patch_h] = np.maximum(
            out[y0:y0 + patch_h, x0:x0 + patch_h],
            cv2.resize(heatmap_pred[i], (patch_w, patch_h)))
    # some post-processing
    threshold = 0.2
    out[out > threshold] = 1.
    out[out < threshold] = 0.
    out = (255. * out).astype(np.uint8)
    # back to BGR for opencv display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(image, 0.4, out, 0.6, 0)
    cv2.imshow('out', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

