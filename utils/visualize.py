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


def visualize_heatmaps(image, heatmaps, vecmaps,
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
        heatmap = np.tile(np.expand_dims(heatmap, axis=2),
                          (1, 1, 3))
        out_img += heatmap * col.reshape((1, 1, 3))
    out_img = cv2.resize(out_img, (img_w, img_h),
                         interpolation=cv2.INTER_NEAREST)
    out_img = (255. * out_img).astype(np.uint8)
    out_img = cv2.addWeighted(out_img, .9, image, 0.1, 0)
    for i, (kp1, kp2) in enumerate(pairs):
        y_indices_1, x_indices_1 = heatmaps[:, :, kp1].nonzero()
        for x, y in zip(x_indices_1, y_indices_1):
            x0, y0 = scale_w * x, scale_h * y
            delta_x = scale_w * int(vecmaps[y, x, 4 * i])
            delta_y = scale_h * int(vecmaps[y, x, 4 * i + 1])
            col = (255. * colors[kp1][:3]).astype(np.uint8)
            col = tuple(map(int, col))
            out_img = cv2.line(out_img, (x0, y0),
                               (x0 + delta_x, y0 + delta_y),
                               col, 1)
        y_indices_2, x_indices_2 = heatmaps[:, :, kp2].nonzero()
        for x, y in zip(x_indices_2, y_indices_2):
            x0, y0 = scale_w * x, scale_h * y
            delta_x = scale_w * int(vecmaps[y, x, 4 * i + 2])
            delta_y = scale_h * int(vecmaps[y, x, 4 * i + 3])
            col = (255. * colors[kp2][:3]).astype(np.uint8)
            col = tuple(map(int, col))
            out_img = cv2.line(out_img, (x0, y0),
                               (x0 + delta_x, y0 + delta_y),
                               col, 1)
    return out_img

