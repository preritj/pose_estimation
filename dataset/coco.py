from collections import defaultdict
import numpy as np
from tqdm import tqdm
from pose_data import Pose


class COCO(Pose):
    def __init__(self, image_dir, annotation_files=None):
        super().__init__(image_dir, annotation_files)

    def _build_dataset(self, dataset):
        self.dataset_count += 1
        for img in dataset['images']:
            self.imgs[img['id']] = {'file_name': img['file_name'],
                                    'height': img['height'],
                                    'width': img['width']}

        kp_names = dataset['categories'][0]['keypoints']

        for ann in tqdm(dataset['annotations']):
            if ann['category_id'] != 1:
                continue
            area = ann['area']
            img = self.imgs[ann['image_id']]
            img_area = img['height'] * img['width']
            num_keypoints = ann['num_keypoints']
            if ann['iscrowd'] or (area > .25 * img_area) or (num_keypoints < 2):
                ignore_region = ann['segmentation']
                self.masks[ann['image_id']].append(
                    {'ignore_region': ignore_region})
                continue
            bbox = ann['bbox']
            kp = np.array(ann['keypoints'])
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]
            keypoints = np.zeros((self.num_keypoints, 3))

            for kp_name, x, y, v in zip(kp_names, xs, ys, vs):
                if kp_name not in self.keypoints.keys():
                    continue
                kp_idx = self.keypoints[kp_name]
                keypoints[kp_idx] = [x, y, v]
            # hack for neck in COCO
            if (vs[5] > 0) and (vs[6] > 0):
                keypoints[2] = np.mean(keypoints[3:5], axis=0)
                if vs[0] > 0:
                    keypoints[2] = (keypoints[1] + 3. * keypoints[2]) / 4.
                elif vs[3] > 0 and vs[4] > 0:
                    face_center = np.array([np.mean(xs[3:5]), np.mean(ys[3:5]), 1])
                    keypoints[2] = (face_center + 3. * keypoints[2]) / 4.
                elif vs[3] > 0:
                    face_center = np.array([xs[3], ys[3], 1])
                    keypoints[2] = (face_center + 3. * keypoints[2]) / 4.
                elif vs[4] > 0:
                    face_center = np.array([xs[4], ys[4], 1])
                    keypoints[2] = (face_center + 3. * keypoints[2]) / 4.
                elif vs[11] > 0 and vs[12] > 0:
                    waist_center = np.array([np.mean(xs[11:13]), np.mean(ys[11:13]), 1])
                    keypoints[2] = (5. * keypoints[2] - waist_center) / 4.
                keypoints[2, 2] = np.int(keypoints[2, 2])
            # hack for head top
            face_center = [-1, -1]
            bottom = keypoints[2]
            if vs[3] > 0 and vs[4] > 0:
                face_center = [np.mean(xs[3:5]), np.mean(ys[3:5])]
            elif vs[1] > 0 and vs[2] > 0:
                face_center = [np.mean(xs[1:3]), np.mean(ys[1:3])]
                if vs[0] > 0 and bottom[2] > 0:
                    bottom = (4. * np.array([xs[0], ys[0], vs[0]]) + bottom) / 5.
            elif vs[1] > 0 and vs[3] > 0:
                face_center = [np.mean(xs[[1, 3]]), np.mean(ys[[1, 3]])]
                if vs[0] > 0 and bottom[2] > 0:
                    bottom = (np.array([xs[0], ys[0], vs[0]]) + 2. * bottom) / 3.
            elif vs[2] > 0 and vs[4] > 0:
                face_center = [np.mean(xs[[2, 4]]), np.mean(ys[[2, 4]])]
                if vs[0] > 0 and bottom[2] > 0:
                    bottom = (np.array([xs[0], ys[0], vs[0]]) + 2. * bottom) / 3.
            if bottom[2] > 0 and np.all(np.array(face_center) > 0):
                # estimate using neck
                head_x = 2. * face_center[0] - bottom[0]
                head_y = 2. * face_center[1] - bottom[1]
                keypoints[0] = [head_x, head_y, 1]
            self.anns[ann['image_id']].append({'keypoints': keypoints,
                                               'bbox': bbox})

    def create_index(self):
        # create index
        print('creating index...')
        self.imgs, self.anns, self.masks = {}, defaultdict(list), defaultdict(list)

        for dataset in self.datasets:
            self._build_dataset(dataset)

        print('index created!')
        self.ids = list(self.anns.keys())
