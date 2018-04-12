import multiprocessing
import cv2
import numpy as np
from coco import COCO
from mpii import MPII
from poseTrack import PoseTrack


class PoseDataGenerator(object):
    def __init__(self):
        self.datasets = []

    def add_dataset(self, name, img_dir, ann_dir,
                    annotation_files=None, weight=1.):
        if name == 'coco':
            ds = COCO(img_dir, annotation_files)
        elif name == 'mpii':
            ds = MPII(img_dir, annotation_files)
        elif name == 'posetrack':
            ds = PoseTrack(img_dir, annotation_files)
        else:
            raise RuntimeError('Dataset not supported')

        if annotation_files is None:
            meta = ds.load_annotations(img_dir, ann_dir)
        else:
            meta = ds.save_annotations(ann_dir)

        self.datasets.append({'name': name,
                              'source': meta,
                              'weight': weight,
                              'itr': self._generator(meta)})

    def probs(self):
        probs = [ds['weight'] for ds in self.datasets]
        probs = np.array(probs)
        return probs / np.sum(probs)

    @staticmethod
    def _generator(meta):
        for img_file, ann_file in meta:
            yield (img_file, ann_file)

    def data_gen(self):
        probs = self.probs()
        try:
            while True:
                ds = np.random.choice(self.datasets, p=probs)
                yield next(ds['itr'])
        except StopIteration:
            return








