import os
import numpy as np
from coco import COCO
from mpii import MPII
from poseTrack import PoseTrack
import tensorflow as tf
import functools

slim_example_decoder = tf.contrib.slim.tfexample_decoder


def _decoder():
    keys_to_features = {
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/shape':
            tf.FixedLenFeature([2], tf.int64),
        'image/num_instances':
            tf.FixedLenFeature((), tf.int64),
        'image/person/bbox':
            tf.VarLenFeature(tf.float32),
        'image/person/keypoints':
            tf.VarLenFeature(tf.float32),
        'image/mask/x':
            tf.VarLenFeature(tf.int64),
        'image/mask/y':
            tf.VarLenFeature(tf.int64)
    }
    items_to_handlers = {
        'image': slim_example_decoder.ItemHandlerCallback(
            'image/filename', _image_decoder),
        'mask': slim_example_decoder.ItemHandlerCallback(
            ['image/mask/x', 'image/mask/y', 'image/shape'],
            _mask_decoder),
        'keypoints': slim_example_decoder.ItemHandlerCallback(
            ['image/person/keypoints', 'image/num_instances'],
            _keypoints_decoder)
    }
    decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)
    return decoder


def _image_decoder(keys_to_tensors):
    filename = keys_to_tensors['image/filename']
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    # image_decoded = tf.image.resize_images(image_decoded, size=[300, 300])
    return image_decoded


def _mask_decoder(keys_to_tensors):
    mask_x = keys_to_tensors['image/mask/x']
    mask_y = keys_to_tensors['image/mask/y']
    shape = keys_to_tensors['image/shape']
    if isinstance(shape, tf.SparseTensor):
        shape = tf.sparse_tensor_to_dense(shape)
    if isinstance(mask_x, tf.SparseTensor):
        mask_x = tf.sparse_tensor_to_dense(mask_x)
    if isinstance(mask_y, tf.SparseTensor):
        mask_y = tf.sparse_tensor_to_dense(mask_y)
    indices = tf.stack([mask_x, mask_y], axis=1)
    values = tf.ones_like(mask_x)
    mask = tf.SparseTensor(indices=indices, values=values,
                           dense_shape=shape)
    mask = tf.sparse_tensor_to_dense(mask, default_value=0)
    #mask.set_shape([None, None])
    #mask = tf.image.resize_images(mask, size=[300, 300])
    return mask


def _keypoints_decoder(keys_to_tensor):
    keypoints = keys_to_tensor['image/person/keypoints']
    num_instances = keys_to_tensor['image/num_instances']
    if isinstance(num_instances, tf.SparseTensor):
        num_instances = tf.sparse_tensor_to_dense(num_instances)
    shape = [num_instances] + [15, 3]
    if isinstance(keypoints, tf.SparseTensor):
        keypoints = tf.sparse_reshape(keypoints, shape=shape)
        keypoints = tf.sparse_tensor_to_dense(keypoints)
    else:
        keypoints = tf.reshape(keypoints, shape=shape)
    return keypoints


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

    def read_data(self, train_config):
        probs = self._get_probs()
        probs = tf.cast(probs, tf.float32)
        decoder = _decoder()
        filenames = [ds['tfrecord_path'] for ds in self.datasets]
        file_ids = list(range(len(filenames)))
        dataset = tf.data.Dataset.from_tensor_slices((file_ids, filenames))
        dataset = dataset.apply(tf.contrib.data.rejection_resample(
            class_func=lambda c, _: c,
            target_dist=probs,
            seed=42))
        dataset = dataset.map(lambda _, a: a[1])
        if train_config.shuffle:
            dataset = dataset.shuffle(
                train_config.filenames_shuffle_buffer_size)

        dataset = dataset.repeat(train_config.num_epochs or None)

        file_read_func = functools.partial(tf.data.TFRecordDataset,
                                           buffer_size=8 * 1000 * 1000)
        records_dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                file_read_func, cycle_length=train_config.num_readers,
                block_length=train_config.read_block_length, sloppy=True))
        if train_config.shuffle:
            records_dataset.shuffle(train_config.shuffle_buffer_size)
        # records_dataset = records_dataset.batch(config.batch_size)
        tensor_dataset = records_dataset.map(
            decoder.decode, num_parallel_calls=train_config.num_parallel_map_calls)
        return tensor_dataset.prefetch(train_config.prefetch_size)
