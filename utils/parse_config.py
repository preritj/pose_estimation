import yaml


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
                 batch_size=2):
        self.shuffle = shuffle
        self.filenames_shuffle_buffer_size = filenames_shuffle_buffer_size
        self.num_parallel_map_calls = num_parallel_map_calls
        self.num_epochs = num_epochs
        self.num_readers = num_readers
        self.read_block_length = read_block_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.prefetch_size = prefetch_size
        self.batch_size = batch_size

    def __repr__(self):
        return 'train_config'


class ModelConfig(yaml.YAMLObject):
    yaml_tag = u'!model'

    def __init__(self,
                 input_shape=None,
                 output_shape=None,
                 num_keypoints=15):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_keypoints = num_keypoints

    def __repr__(self):
        return 'model_config'


def parse_config(config_file):
    cfgs = {}
    with open(config_file, 'r') as f:
        for cfg in yaml.load_all(f):
            cfgs[str(cfg)] = cfg
    return cfgs
