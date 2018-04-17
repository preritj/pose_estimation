import os
import numpy as np
from coco import COCO
from mpii import MPII
from poseTrack import PoseTrack
import tensorflow as tf
import functools
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops

slim_example_decoder = tf.contrib.slim.tfexample_decoder


class Tensor(slim_example_decoder.ItemHandler):
    """An ItemHandler that returns a parsed Tensor."""

    def __init__(self, tensor_key, num_instances_key=None, shape=None, default_value=0):
        """Initializes the Tensor handler.
        Tensors are, by default, returned without any reshaping. However, there are
        two mechanisms which allow reshaping to occur at load time. If `shape_keys`
        is provided, both the `Tensor` corresponding to `tensor_key` and
        `shape_keys` is loaded and the former `Tensor` is reshaped with the values
        of the latter. Alternatively, if a fixed `shape` is provided, the `Tensor`
        corresponding to `tensor_key` is loaded and reshape appropriately.
        If neither `shape_keys` nor `shape` are provided, the `Tensor` will be
        returned without any reshaping.
        Args:
          tensor_key: the name of the `TFExample` feature to read the tensor from.
          shape_keys: Optional name or list of names of the TF-Example feature in
            which the tensor shape is stored. If a list, then each corresponds to
            one dimension of the shape.
          shape: Optional output shape of the `Tensor`. If provided, the `Tensor` is
            reshaped accordingly.
          default_value: The value used when the `tensor_key` is not found in a
            particular `TFExample`.
        Raises:
          ValueError: if both `shape_keys` and `shape` are specified.
        """
        self._tensor_key = tensor_key
        self._num_instances_key = num_instances_key
        self._shape = shape
        self._default_value = default_value
        keys = [tensor_key]
        if num_instances_key is not None:
            keys.append(num_instances_key)
        super(Tensor, self).__init__(keys)

    def tensors_to_item(self, keys_to_tensors):
        tensor = keys_to_tensors[self._tensor_key]
        shape = self._shape
        shape_dims = shape if self._shape is not None else []
        if self._num_instances_key is not None:
            shape_dim = keys_to_tensors[self._num_instances_key]
            if isinstance(shape_dim, sparse_tensor.SparseTensor):
                shape_dim = sparse_ops.sparse_tensor_to_dense(shape_dim)
            shape_dims = shape_dims + [shape_dim]
            shape = array_ops.reshape(array_ops.stack(shape_dims), [-1])
        tensor = array_ops.reshape(tensor, shape)
        return tensor


def _decoder():
    keys_to_features = {
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/shape':
            tf.FixedLenFeature([2], tf.int64),
        'image/num_instances':
            tf.FixedLenFeature((), tf.int64),
        'image/person/bbox':
            tf.VarLenFeature(tf.string),
        'image/person/keypoints':
            tf.VarLenFeature(tf.float32),
        'image/mask':
            tf.FixedLenFeature((), tf.string)
    }
    items_to_handlers = {
        'image': slim_example_decoder.ItemHandlerCallback(
            'image/filename', _image_decoder),
        'mask': slim_example_decoder.ItemHandlerCallback(
            'image/mask', _mask_decoder),
        'keypoints': Tensor('image/person/keypoints', 'image/num_instances',
                            shape=[15, 3])
    }
    decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)
    return decoder


def _image_decoder(keys_to_tensors):
    filename = keys_to_tensors['image/filename']
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.image.resize_images(image_decoded, size=[300, 300])
    return image_decoded


def _mask_decoder(keys_to_tensors):
    mask = keys_to_tensors['image/mask']
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize_images(mask, size=[300, 300])
    return tf.squeeze(mask)


def _keypoints_decoder(keys_to_tensor):
    keypoints = keys_to_tensor['image/person/keypoints']
    # keypoints = tf.decode_raw(keypoints, out_type=tf.float32)
    return tf.squeeze(keypoints)


class PoseDataReader(object):
    def __init__(self):
        self.datasets = []

    def add_dataset(self, name, img_dir, tfrecord_path,
                    annotation_files=None, weight=1.,
                    overwrite_tfrecords=False):
        tfrecord_dir = os.path.dirname(tfrecord_path)
        if not os.path.exists(tfrecord_dir):
            os.makedirs(tfrecord_dir)
            overwrite_tfrecords = True
        if overwrite_tfrecords:
            if name == 'coco':
                ds = COCO(img_dir, annotation_files)
            elif name == 'mpii':
                ds = MPII(img_dir, annotation_files)
            elif name == 'posetrack':
                ds = PoseTrack(img_dir, annotation_files)
            else:
                raise RuntimeError('Dataset not supported')
            ds.create_tf_record(tfrecord_path)
        assert os.path.exists(tfrecord_path), \
            "{} does not exist".format(tfrecord_path)

        self.datasets.append({'name': name,
                              'tfrecord_path': tfrecord_path,
                              'weight': weight})

    def _get_probs(self):
        probs = [ds['weight'] for ds in self.datasets]
        probs = np.array(probs)
        return probs / np.sum(probs)

    def read_data(self, config):
        probs = self._get_probs()
        probs = tf.cast(probs, tf.float32)
        decoder = _decoder()
        filenames = [ds['tfrecord_path'] for ds in self.datasets]
        file_ids = tf.constant(list(range(len(filenames))))
        filenames = tf.constant(filenames)
        dataset = tf.data.Dataset.from_tensor_slices((filenames, file_ids))
        # dataset = dataset.apply(tf.contrib.data.rejection_resample(
        #     class_func=lambda _, c: c,
        #     target_dist=probs,
        #     seed=42))
        dataset = dataset.map(lambda a, _: a)
        if config.shuffle:
            dataset = dataset.shuffle(
                config.filenames_shuffle_buffer_size)

        dataset = dataset.repeat(config.num_epochs or None)

        file_read_func = functools.partial(tf.data.TFRecordDataset,
                                           buffer_size=8 * 1000 * 1000)
        records_dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                file_read_func, cycle_length=config.num_readers,
                block_length=config.read_block_length, sloppy=True))
        if config.shuffle:
            records_dataset.shuffle(config.shuffle_buffer_size)
        # records_dataset = records_dataset.batch(config.batch_size)
        tensor_dataset = records_dataset.map(
            decoder.decode, num_parallel_calls=config.num_parallel_map_calls)
        return tensor_dataset.prefetch(config.prefetch_size)
