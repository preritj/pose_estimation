from tensorpack import RandomChooseData
from coco import COCO
from mpii import MPII
from poseTrack import PoseTrack
from pose_data import PoseMetadata


class PoseDataFlow(RandomChooseData):
    def __init__(self, datasets):
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

        super().__init__(df_lists)





