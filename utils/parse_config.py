import yaml


class DataConfig(yaml.YAMLObject):
    yaml_tag = u'!data'

    def __init__(self,
                 datasets,
                 keypoints,
                 skeleton,
                 sigma=8):
        assert len(datasets) > 0, "Specify datasets"
        self.datasets = datasets
        self.keypoints = keypoints
        self.skeleton = skeleton
        self.sigma = sigma

    def __repr__(self):
        return 'data_config'


class TrainConfig(yaml.YAMLObject):
    yaml_tag = u'!train'

    def __init__(self,
                 shuffle=True,
                 filenames_shuffle_buffer_size=100,
                 num_parallel_map_calls=2,
                 num_epochs=0,
                 num_readers=32,
                 read_block_length=32,
                 shuffle_buffer_size=2048,
                 prefetch_size=512,
                 batch_size=2,
                 learning_rate=0.001,
                 optimizer=None,
                 augmentation=None,
                 preprocess=None):
        self.shuffle = shuffle
        self.filenames_shuffle_buffer_size = filenames_shuffle_buffer_size
        self.num_parallel_map_calls = num_parallel_map_calls
        self.num_epochs = num_epochs
        self.num_readers = num_readers
        self.read_block_length = read_block_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if optimizer is None:
            optimizer = {'name': 'adam'}
        self.optimizer = optimizer
        self.augmentation = augmentation
        self.preprocess = preprocess

    def __repr__(self):
        return 'train_config'


class ModelConfig(yaml.YAMLObject):
    yaml_tag = u'!model'

    def __init__(self,
                 input_shape=None,
                 output_shape=None,
                 keypoints=None,
                 depth_multiplier=1.,
                 min_depth=8,
                 skip_layers=None,
                 fpn_depth=96):
        if input_shape is None:
            input_shape = [224, 224]
        if output_shape is None:
            output_shape = input_shape
        self.input_shape = input_shape
        self.output_shape = output_shape
        if keypoints is None:
            keypoints = []
        self.keypoints = keypoints
        self.depth_multiplier = depth_multiplier
        self.min_depth = min_depth
        if skip_layers is None:
            skip_layers = []
        self.skip_layers = skip_layers
        self.fpn_depth = fpn_depth

    def __repr__(self):
        return 'model_config'


def parse_config(config_file):
    cfgs = {}
    with open(config_file, 'r') as f:
        for cfg in yaml.load_all(f):
            cfgs[str(cfg)] = cfg
    return cfgs
