import numpy as np
import cv2
import matplotlib.cm as cm
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw


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
            x1 = np.clip(np.around(x + offsetmaps[y, x, i]), 0, w - 1).astype(np.uint8)
            y1 = np.clip(np.around(y + offsetmaps[y, x, num_keypoints + i]), 0, h - 1).astype(np.uint8)
            new_heatmap[y1, x1] = 1.
        heatmaps[:, :, i] = new_heatmap
        heatmap = np.tile(np.expand_dims(new_heatmap, axis=2),
                          (1, 1, 3))
        out_img += heatmap * col.reshape((1, 1, 3))
    out_img = cv2.resize(out_img, (img_w, img_h),
                         interpolation=cv2.INTER_NEAREST)
    out_img = (255. * out_img).astype(np.uint8)
    out_img = cv2.addWeighted(out_img, .6, image, 0.4, 0)
    for i, (kp1, kp2) in enumerate(pairs):
        y_indices_1, x_indices_1 = heatmaps[:, :, kp1].nonzero()
        for x, y in zip(x_indices_1, y_indices_1):
            x1, y1 = vecmaps[y, x, 4 * i], vecmaps[y, x, 4 * i + 1]
            x1 = np.around(x + x1).astype(np.uint8)
            y1 = np.around(y + y1).astype(np.uint8)
            x0 = int(scale_w * (x + 0.5))
            y0 = int(scale_h * (y + 0.5))
            if (x1 > 0) and (x1 < w - 1) and (y1 > 0) and (y1 < h - 1):
                delta_x = int(scale_w * (
                    vecmaps[y, x, 4 * i] + offsetmaps[y1, x1, kp2]))
                delta_y = int(scale_h * (vecmaps[y, x, 4 * i + 1]
                                         + offsetmaps[y1, x1, num_keypoints + kp2]))
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
            x1 = np.around(x + x1).astype(np.uint8)
            y1 = np.around(y + y1).astype(np.uint8)
            x0 = int(scale_w * (x + 0.5))
            y0 = int(scale_h * (y + 0.5))
            if (x1 > 0) and (x1 < w - 1) and (y1 > 0) and (y1 < h - 1):
                delta_x = int(scale_w * (
                    vecmaps[y, x, 4 * i + 2] + offsetmaps[y1, x1, kp1]))
                delta_y = int(scale_h * (vecmaps[y, x, 4 * i + 3]
                                         + offsetmaps[y1, x1, num_keypoints + kp1]))
            else:
                delta_x = int(scale_w * (vecmaps[y, x, 4 * i + 2]))
                delta_y = int(scale_h * (vecmaps[y, x, 4 * i + 3]))
            col = (255. * colors[kp2][:3]).astype(np.uint8)
            col = tuple(map(int, col))
            out_img = cv2.line(out_img, (x0, y0),
                               (x0 + delta_x, y0 + delta_y),
                               col, 1)
    return out_img

