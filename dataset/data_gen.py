import os
import numpy as np
from coco import COCO
from mpii import MPII
from poseTrack import PoseTrack
import tensorflow as tf
import functools

slim_example_decoder = tf.contrib.slim.tfexample_decoder


class PoseDataReader(object):
    def __init__(self, num_keypoints=15):
        self.datasets = []
        self.num_keypoints = num_keypoints

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

    @staticmethod
    def _get_tensor(tensor):
        if isinstance(tensor, tf.SparseTensor):
            return tf.sparse_tensor_to_dense(tensor)
        return tensor

    @staticmethod
    def _image_decoder(keys_to_tensors):
        filename = keys_to_tensors['image/filename']
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        return image_decoded

    @staticmethod
    def _mask_decoder(keys_to_tensors):
        mask_x = PoseDataReader._get_tensor(
            keys_to_tensors['image/mask/x'])
        mask_y = PoseDataReader._get_tensor(
            keys_to_tensors['image/mask/y'])
        shape = PoseDataReader._get_tensor(
            keys_to_tensors['image/shape'])
        indices = tf.stack([mask_x, mask_y], axis=1)
        values = tf.ones_like(mask_x)
        mask = tf.SparseTensor(indices=indices, values=values,
                               dense_shape=shape)
        mask = tf.sparse_tensor_to_dense(mask, default_value=0)
        return mask

    @staticmethod
    def _keypoints_decoder(keys_to_tensor, num_keypoints=15):
        keypoints = keys_to_tensor['image/person/keypoints']
        num_instances = PoseDataReader._get_tensor(
            keys_to_tensor['image/num_instances'])
        shape = [num_instances] + [num_keypoints, 3]
        if isinstance(keypoints, tf.SparseTensor):
            keypoints = tf.sparse_reshape(keypoints, shape=shape)
            keypoints = tf.sparse_tensor_to_dense(keypoints)
        else:
            keypoints = tf.reshape(keypoints, shape=shape)
        return keypoints

    @staticmethod
    def _bbox_decoder(keys_to_tensor):
        bbox = keys_to_tensor['image/person/bbox']
        num_instances = PoseDataReader._get_tensor(
            keys_to_tensor['image/num_instances'])
        shape = [num_instances] + [4]
        if isinstance(bbox, tf.SparseTensor):
            bbox = tf.sparse_reshape(bbox, shape=shape)
            bbox = tf.sparse_tensor_to_dense(bbox)
        else:
            bbox = tf.reshape(bbox, shape=shape)
        return bbox

    def _decoder(self):
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
        keypoints_decoder = functools.partial(
            self._keypoints_decoder, num_keypoints=self.num_keypoints)
        items_to_handlers = {
            'image': slim_example_decoder.ItemHandlerCallback(
                'image/filename', self._image_decoder),
            'mask': slim_example_decoder.ItemHandlerCallback(
                ['image/mask/x', 'image/mask/y', 'image/shape'],
                self._mask_decoder),
            'keypoints': slim_example_decoder.ItemHandlerCallback(
                ['image/person/keypoints', 'image/num_instances'],
                keypoints_decoder),
            'bbox': slim_example_decoder.ItemHandlerCallback(
                ['image/person/bbox', 'image/num_instances'],
                self._bbox_decoder)
        }
        decoder = slim_example_decoder.TFExampleDecoder(keys_to_features,
                                                        items_to_handlers)
        return decoder

    def read_data(self, train_config):
        probs = self._get_probs()
        probs = tf.cast(probs, tf.float32)
        decoder = self._decoder()
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
        decode_fn = functools.partial(
            decoder.decode, items=['image', 'keypoints', 'bbox', 'mask'])
        tensor_dataset = records_dataset.map(
            decode_fn, num_parallel_calls=train_config.num_parallel_map_calls)
        return tensor_dataset.prefetch(train_config.prefetch_size)
