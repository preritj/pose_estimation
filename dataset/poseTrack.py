from collections import defaultdict
import numpy as np
from tqdm import tqdm
from pose_data import Pose


class PoseTrack(Pose):
    def __init__(self, image_dir, annotation_files=None):
        super().__init__(image_dir, annotation_files)

    def _build_dataset(self, dataset, kp_dict):
        self.dataset_count += 1
        for annotations in dataset['annolist']:
            img_name = annotations['image'][0]['name']
            img_path = os.path.join(self.image_dir, img_name)
            im = Image.open(img_path)
            width, height = im.size
            img_id = self.dataset_count*10000 + annotations['imgnum'][0]
            self.imgs[img_id] = {'file_name': img_name,
                                 'height': height,
                                 'width': width}
            if not annotations['is_labeled']:
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
                self.anns[img_id].append({'keypoints': keypoints_clean})

            if 'ignore_regions' in annotations.keys():
                ignore_regions = annotations['ignore_regions']
                polygons = []
                for reg in ignore_regions:
                    polygon = []
                    polygon_pts = reg['point']
                    for point in polygon_pts:
                        polygon.append(point['x'][0])
                        polygon.append(point['y'][0])
                    polygons.append(polygon)
                self.masks[img_id].append({'ignore_region': polygons})

    def create_index(self):
        kp_dict = {0: 'right_ankle',
                   1: 'right_knee',
                   2: 'right_hip',
                   3: 'left_hip',
                   4: 'left_knee',
                   5: 'left_ankle',
                   6: 'right_wrist',
                   7: 'right_elbow',
                   8: 'right_shoulder',
                   9: 'left_shoulder',
                   10: 'left_elbow',
                   11: 'left_wrist',
                   12: 'neck',
                   13: 'nose',
                   14: 'head'}
        # create index
        print('creating index...')
        self.imgs, self.anns, self.masks = {}, defaultdict(list), defaultdict(list)

        for dataset in tqdm(self.datasets):
            self._build_dataset(dataset, kp_dict)

        print('index created!')
        self.ids = list(self.anns.keys())
