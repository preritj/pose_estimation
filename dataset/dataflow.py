import multiprocessing
import cv2
from tensorpack import (RandomChooseData, PrefetchDataZMQ,
                        MapData, MultiThreadMapData, BatchData)
from coco import COCO
from mpii import MPII
from poseTrack import PoseTrack


def get_dataflow(datasets, batch_size, out_shape, nr_thread=None):
    """Takes as input dictionary of datasets"""
    df_lists = []
    for name, data in datasets:
        if name == 'coco':
            df = COCO(data['images'], data['annotations'])
            df_lists.append((df, data['prob']))
        elif name == 'mpii':
            df = MPII(data['images'], data['annotations'])
            df_lists.append((df, data['prob']))
        elif name == 'posetrack':
            df = PoseTrack(data['images'], data['annotations'])
            df_lists.append((df, data['prob']))
        else:
            raise RuntimeError('Dataset not supported')

    if len(df_lists) == 1:
        df = df_lists[0]
    else:
        df = RandomChooseData(df_lists)

    def mapf(meta):
        fname = meta.img_path
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        heatmap = meta.get_heatmap(out_shape)
        return [image, heatmap]
    if nr_thread is None:
        nr_thread = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading

    df = MapData(df, mapf)
    df = PrefetchDataZMQ(df, nr_thread)
    df = BatchData(df, batch_size, use_list=True)
    return df







