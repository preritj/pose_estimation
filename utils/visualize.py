import numpy as np
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
                   (right, top), (left, top)], width=3, fill='red')
    np.copyto(image, np.array(image_pil))
    return image


def visualize_heatmaps(heatmaps):
    h, w, num_keypoints = heatmaps.shape
    out_img = np.zeros((h, w, 3))
    colors = cm.jet(np.linspace(0, 1, num_keypoints))
    for i in range(num_keypoints):
        col = colors[i][:3]
        heatmap = heatmaps[:, :, i]
        heatmap = np.tile(np.expand_dims(heatmap, axis=2),
                          (1, 1, 3))
        out_img += heatmap * col.reshape((1, 1, 3))
    return out_img.astype(np.float32)

