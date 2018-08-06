import numpy as np
import cv2
import matplotlib.cm as cm
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
from collections import defaultdict


def visualize_bboxes_on_image(image, boxes):
    image_pil = Image.fromarray(image)
    for box in boxes:
        draw = ImageDraw.Draw(image_pil)
        ymin, xmin, ymax, xmax = box
        im_width, im_height = image_pil.size
        left, right, top, bottom = (xmin * im_width,
                                    xmax * im_width,
                                    ymin * im_height,
                                    ymax * im_height)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=2, fill='red')
    np.copyto(image, np.array(image_pil))
    return image


def visualize_heatmaps(image, heatmaps, vecmaps, offsetmaps,
                       pairs=([0, 1], [2, 1]), threshold=0.2):
    heatmaps[heatmaps > threshold] = 1.
    heatmaps[heatmaps <= threshold] = 0.
    img_h, img_w, _ = image.shape
    h, w, num_keypoints = heatmaps.shape
    scale_h, scale_w = int(img_h / h), int(img_w / w)
    out_img = np.zeros((h, w, 3))
    colors = cm.jet(np.linspace(0, 1, num_keypoints))
    for i in range(num_keypoints):
        col = colors[i][:3]
        heatmap = heatmaps[:, :, i]
        new_heatmap = np.zeros_like(heatmap)
        y_indices, x_indices = heatmap.nonzero()
        for x, y in zip(x_indices, y_indices):
            x1 = np.clip(np.floor(x + 0.5 + offsetmaps[y, x, 2 * i]), 0, w - 1).astype(np.uint8)
            y1 = np.clip(np.floor(y + 0.5 + offsetmaps[y, x, 2 * i + 1]), 0, h - 1).astype(np.uint8)
            new_heatmap[y1, x1] = 1.
        heatmaps[:, :, i] = new_heatmap
        heatmap = np.tile(np.expand_dims(new_heatmap, axis=2),
                          (1, 1, 3))
        out_img += heatmap * col.reshape((1, 1, 3))
    out_img = cv2.resize(out_img, (img_w, img_h),
                         interpolation=cv2.INTER_NEAREST)
    out_img = (255. * out_img).astype(np.uint8)
    out_img = cv2.addWeighted(out_img, 0.4, image, 0.6, 0)
    for i, (kp1, kp2) in enumerate(pairs):
        y_indices_1, x_indices_1 = heatmaps[:, :, kp1].nonzero()
        for x, y in zip(x_indices_1, y_indices_1):
            x1, y1 = vecmaps[y, x, 4 * i], vecmaps[y, x, 4 * i + 1]
            x1 = np.floor(x + 0.5 + x1).astype(np.int16)
            y1 = np.floor(y + 0.5 + y1).astype(np.int16)
            x0 = int(scale_w * (x + 0.5))
            y0 = int(scale_h * (y + 0.5))
            if (x1 > 0) and (x1 < w - 1) and (y1 > 0) and (y1 < h - 1):
                delta_x = int(scale_w * (x1 - x + offsetmaps[y1, x1, 2 * kp2]))
                delta_y = int(scale_h * (y1 - y + offsetmaps[y1, x1, 2 * kp2 + 1]))
            else:
                delta_x = int(scale_w * (vecmaps[y, x, 4 * i]))
                delta_y = int(scale_h * (vecmaps[y, x, 4 * i + 1]))
            col = (255. * colors[kp1][:3]).astype(np.uint8)
            col = tuple(map(int, col))
            out_img = cv2.line(out_img, (x0, y0),
                               (x0 + delta_x, y0 + delta_y),
                               col, 1)
        y_indices_2, x_indices_2 = heatmaps[:, :, kp2].nonzero()
        for x, y in zip(x_indices_2, y_indices_2):
            x1, y1 = vecmaps[y, x, 4 * i + 2], vecmaps[y, x, 4 * i + 3]
            x1 = np.floor(x + 0.5 + x1).astype(np.int16)
            y1 = np.floor(y + 0.5 + y1).astype(np.int16)
            x0 = int(scale_w * (x + 0.5))
            y0 = int(scale_h * (y + 0.5))
            if (x1 > 0) and (x1 < w - 1) and (y1 > 0) and (y1 < h - 1):
                delta_x = int(scale_w * (x1 - x + offsetmaps[y1, x1, 2 * kp1]))
                delta_y = int(scale_h * (y1 - y + offsetmaps[y1, x1, 2 * kp1 + 1]))
            else:
                delta_x = int(scale_w * (vecmaps[y, x, 4 * i + 2]))
                delta_y = int(scale_h * (vecmaps[y, x, 4 * i + 3]))
            col = (255. * colors[kp2][:3]).astype(np.uint8)
            col = tuple(map(int, col))
            out_img = cv2.line(out_img, (x0, y0),
                               (x0 + delta_x, y0 + delta_y),
                               col, 1)
    return out_img


def visualize_instances(image, heatmaps, vecmaps, offsetmaps,
                        pairs=([0, 1], [2, 1]), threshold=0.1):
    persons = greedy_connect(heatmaps, vecmaps, offsetmaps,
                             pairs, threshold)
    colors = cm.hsv(np.linspace(0, 1, len(persons) + 1))
    out_img = np.zeros_like(image, dtype=np.uint8)
    out_img = cv2.addWeighted(out_img, 0.4, image, 0.6, 0)
    img_h, img_w, _ = image.shape
    h, w, num_keypoints = heatmaps.shape
    scale_h, scale_w = int(img_h / h), int(img_w / w)
    for i, person in enumerate(persons):
        col = colors[i]
        col = col[:3]
        col = (255. * col).astype(np.uint8)
        col = tuple(map(int, col))
        for kp1, kp2 in pairs:
            x1, y1, v1 = person[kp1]
            x2, y2, v2 = person[kp2]
            if (v1 == 0) and (v2 == 0):
                continue
            x1, y1 = int(scale_w * (x1 + 0.5)), int(scale_h * (y1 + 0.5))
            x2, y2 = int(scale_w * (x2 + 0.5)), int(scale_h * (y2 + 0.5))
            out_img = cv2.line(out_img, (x1, y1),
                               (x2, y2), col, 1)
            out_img[max(y1 - 3, 0): min(y1 + 3, img_h),
                    max(x1 - 3, 0): min(x1 + 3, img_w)] = col
            out_img[max(y2 - 3, 0): min(y2 + 3, img_h),
                    max(x2 - 3, 0): min(x2 + 3, img_w)] = col
    # out_img = cv2.addWeighted(out_img, .5, image, 0.5, 0)
    return out_img


def greedy_connect(heatmaps, vecmaps, offsetmaps,
                   pairs=([0, 1], [2, 1]), threshold=0.1):
    keypoints_graph = defaultdict(list)
    vecmap_indices = {}
    for i, (kp1, kp2) in enumerate(pairs):
        keypoints_graph[kp1].append(kp2)
        keypoints_graph[kp2].append(kp1)
        vecmap_indices[(kp1, kp2)] = (4 * i, 4 * i + 1)
        vecmap_indices[(kp2, kp1)] = (4 * i + 2, 4 * i + 3)
    h, w, num_keypoints = heatmaps.shape
    y_indices, x_indices, kp_indices = np.where(heatmaps > threshold)
    scores = heatmaps[y_indices, x_indices, kp_indices]
    ordering = np.argsort(scores)
    keypoints = np.stack((x_indices, y_indices, kp_indices), axis=-1)
    keypoints = list(keypoints[ordering])
    persons = []

    def _travel_graph(seed_, person_):
        x1, y1, kp1 = seed_
        person_[kp1] = (x1, y1, 0)
        for kp2 in keypoints_graph[kp1]:
            if kp2 in person_.keys():
                continue
            x_idx, y_idx = vecmap_indices[(kp1, kp2)]
            dx, dy = vecmaps[y1, x1, x_idx], vecmaps[y1, x1, y_idx]
            x2_ = np.floor(x1 + 0.5 + dx).astype(np.int16)
            y2_ = np.floor(y1 + 0.5 + dy).astype(np.int16)
            if (x2_ > 0) and (x2_ < w - 1) and (y2_ > 0) and (y2_ < h - 1):
                x2 = np.floor(x2_ + offsetmaps[y2_, x2_, 2 * kp2]).astype(np.int16)
                y2 = np.floor(y2_ + offsetmaps[y2_, x2_, 2 * kp2 + 1]).astype(np.int16)
            else:
                x2 = x2_
                y2 = y2_
            x2 = np.clip(x2, 0, w - 1)
            y2 = np.clip(y2, 0, h - 1)
            new_seed_ = (x2, y2, kp2)
            _travel_graph(new_seed_, person_)

    while len(keypoints) > 0:
        x, y, kp = keypoints.pop()
        already_exists = False
        for person in persons:
            x1, y1, v1 = person[kp]
            # person keypoint already matched
            if v1 > 0.5:
                continue
            # person keypoint unmatched and in proximity
            if (abs(x - x1) < 3) and (abs(y - y1) < 3):
                already_exists = True
                person[kp] = (x, y, 1)  # set visibility to 1
                break
        if not already_exists:
            new_person = {}
            # x, y = np.floor(x), np.floor(y)
            start_seed = (x, y, kp)
            _travel_graph(start_seed, new_person)
            new_person[kp] = (x, y, 1)
            persons.append(dict(new_person))

    persons_final = []
    for person in persons:
        n_vis = sum([vis for _, _, vis in person.values()])
        if n_vis > 3:
            persons_final.append(person)
    return persons_final




