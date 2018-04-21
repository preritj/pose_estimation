from collections import defaultdict
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from pose_data import PoseData


class AVAretail(PoseData):
    def __init__(self, pose_cfg, image_dir, annotation_files=None):
        self.idx_count = 0
        self.idx_map = {}
        super().__init__(pose_cfg, image_dir, annotation_files)

    def _build_dataset(self, dataset):
        for i, annotations in tqdm(enumerate(dataset)):
            img_id = i + self.idx_count
            self.idx_count += 1
            img_name = annotations['FileName']
            img_path = os.path.join(self.image_dir, img_name)
            im = Image.open(img_path)
            width, height = im.size
            self.imgs[img_id] = {'filename': img_name,
                                 'shape': [height, width]}
            persons = annotations['Persons']
            if len(persons) == 0:
                continue
            all_keypoints = [person['Keypoints'] for person in persons
                             if 'Keypoints' in person.keys()]
            if len(all_keypoints) == 0:
                continue
            ignore_patches = []
            for person in persons:
                bbox = None
                if 'Bbox' in person.keys():
                    xmin, ymin, w, h = person['Bbox']
                    xmax = int(min(xmin + w, width - 1))
                    ymax = int(min(ymin + h, height - 1))
                    xmin, ymin = int(max(0, xmin)), int(max(0, ymin))
                    bbox = [ymin, xmin, ymax, xmax]
                if 'Keypoints' not in person.keys():
                    # TODO: make mask using bbox
                    if bbox is not None:
                        ymin, xmin, ymax, xmax = bbox
                        patch_pts = [xmin, ymin,
                                     xmax, ymin,
                                     xmax, ymax,
                                     xmin, ymax]
                        ignore_patches.append(patch_pts)
                    continue
                keypoints = np.array(person['Keypoints'])
                if bbox is None:
                    xmin, ymin = np.min(keypoints[:, :2], axis=0)
                    xmax, ymax = np.max(keypoints[:, :2], axis=0)
                    xmax = int(min(xmax, width - 1))
                    ymax = int(min(ymax, height - 1))
                    xmin, ymin = int(max(0, xmin)), int(max(0, ymin))
                    bbox = [ymin, xmin, ymax, xmax]
                self.anns[img_id].append({'keypoints': keypoints,
                                          'bbox': bbox})
            if len(ignore_patches) > 0:
                self.masks[img_id].append({'ignore_region': ignore_patches})

    def create_index(self):
        # create index
        print('creating index...')
        self.imgs, self.anns, self.masks = {}, defaultdict(list), defaultdict(list)

        for dataset in self.datasets:
            self._build_dataset(dataset)

        print('index created!')
        self.ids = list(self.anns.keys())
