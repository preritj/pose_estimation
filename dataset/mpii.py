from collections import defaultdict
import numpy as np
from pose_data import PoseData


class MPII(PoseData):
    def __init__(self, image_dir, annotation_files=None):
        super().__init__(image_dir, annotation_files)

    def _build_dataset(self, dataset, kp_dict):
        self.dataset_count += 1
        dataset = dataset['RELEASE'][0]
        is_train_list = dataset['img_train']

        for i, (is_train, annotations) in enumerate(
                zip(is_train_list, dataset['annolist'])):
            img_name = annotations['image'][0]['name']
            img_id = self.dataset_count*50000 + i
            self.imgs[i] = {'file_name': img_name,
                            'height': None,
                            'width': None}
            if not is_train:
                continue
            persons = annotations['annorect']
            for person in persons:
                if 'annopoints' not in person.keys():
                    continue
                if len(person['annopoints']) == 0:
                    continue
                keypoints = person['annopoints'][0]['point']
                keypoints_clean = np.zeros((self.num_keypoints, 3))
                for kp in keypoints:
                    if 'is_visible' not in kp.keys():
                        kp['is_visible'] = [-1]
                    id_, x, y, v = (kp['id'][0], kp['x'][0],
                                    kp['y'][0], kp['is_visible'])
                    v = v[0] if len(v) > 0 else 2
                    kp_name = kp_dict[id_]
                    if kp_name in self.keypoints.keys():
                        kp_idx = self.keypoints[kp_name]
                        keypoints_clean[kp_idx] = [x, y, v]
                # hack for nose in MPII
                if keypoints_clean[0, 2] > 0 and keypoints_clean[2, 2] > 0:
                    keypoints_clean[1] = np.mean(keypoints_clean[[0, 2], :], axis=0)
                self.anns[i].append({'keypoints': keypoints_clean})

    def create_index(self):
        kp_dict = {0: 'right_ankle',
                   1: 'right_knee',
                   2: 'right_hip',
                   3: 'left_hip',
                   4: 'left_knee',
                   5: 'left_ankle',
                   6: 'pelvis',
                   7: 'thorax',
                   8: 'neck',
                   9: 'head',
                   10: 'right_wrist',
                   11: 'right_elbow',
                   12: 'right_shoulder',
                   13: 'left_shoulder',
                   14: 'left_elbow',
                   15: 'left_wrist'}
        # create index
        print('creating index...')
        self.imgs, self.anns, self.masks = {}, defaultdict(list), defaultdict(list)

        for dataset in self.datasets:
            self._build_dataset(dataset, kp_dict)

        print('index created!')
        self.ids = list(self.anns.keys())