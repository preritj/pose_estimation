import os
import numpy as np
from coco import COCO
from mpii import MPII
from poseTrack import PoseTrack
import tensorflow as tf


class PoseDataReader(object):
    def __init__(self):
        self.datasets = []

    def add_dataset(self, name, img_dir, tfrecord_dir,
                    annotation_files=None, weight=1.,
                    overwrite_tfrecords=False):
        if not os.path.exists(tfrecord_dir):
            os.makedirs(tfrecord_dir)
            overwrite_tfrecords = True
        tfrecord_path = os.path.join(tfrecord_dir, name + '.record')
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
                              'tfrecord_path': tfrecord_dir,
                              'weight': weight})

    def _get_probs(self):
        probs = [ds['weight'] for ds in self.datasets]
        probs = np.array(probs)
        return probs / np.sum(probs)

    def read_data(self, file_read_func, decode_func, config):
        probs = self._get_probs()
        filenames = [ds['tfrecord_path'] for ds in self.datasets]
        file_ids = list(range(len(filenames)))
        dataset = tf.data.Dataset.from_tensor_slices((filenames, file_ids))
        dataset = dataset.apply(tf.contrib.data.rejection_resample(
            class_func=lambda _, c: c,
            target_dist=probs,
            seed=42))
        dataset = dataset.map(lambda a, _: a)
        if config.shuffle:
            dataset = dataset.shuffle(
                config.filenames_shuffle_buffer_size)

        dataset = dataset.repeat(config.num_epochs or None)

        records_dataset = dataset.apply(
            tf.contrib.data.parallel_interleave(
                file_read_func, cycle_length=config.num_readers,
                block_length=config.read_block_length, sloppy=True))
        if config.shuffle:
            records_dataset.shuffle(config.shuffle_buffer_size)
        tensor_dataset = records_dataset.map(
            decode_func, num_parallel_calls=config.num_parallel_map_calls)
        return tensor_dataset.prefetch(config.prefetch_size)
