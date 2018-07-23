import os
import argparse
import tensorflow as tf
from model.mobilenet_pose import MobilenetPose
import functools
from dataset.data_reader import PoseDataReader
from utils.parse_config import parse_config
from utils.dataset_util import keypoints_to_heatmaps_and_vectors
from utils.ops import non_max_suppression
import utils.visualize as vis
from tensorflow.python import pywrap_tensorflow
try:
    import horovod.tensorflow as hvd
    print("Found horovod module, will use distributed training")
    use_hvd = True
except ImportError:
    print("Horovod module not found, will not use distributed training")
    use_hvd = False

slim = tf.contrib.slim
DEBUG = False


class Trainer(object):

    def __init__(self, cfg_file):
        # Define model parameters
        cfg = parse_config(cfg_file)
        self.data_cfg = cfg['data_config']
        self.train_cfg = cfg['train_config']
        self.model_cfg = cfg['model_config']
        self.infer_cfg = cfg['infer_config']
        self.hparams = tf.contrib.training.HParams(
            **self.model_cfg.__dict__,
            num_keypoints=len(self.train_cfg.train_keypoints),
            num_vecs=4 * len(self.train_cfg.train_skeletons))
        kp_dict = {kp: i for i, kp in
                   enumerate(self.train_cfg.train_keypoints)}
        self.pairs = [[kp_dict[kp1], kp_dict[kp2]] for kp1, kp2
                      in self.train_cfg.train_skeletons]
        self.cpu_device = '/cpu:0'
        self.param_server_device = '/gpu:0'

    def get_features_labels_data(self):
        """returns dataset containing (features, labels)"""
        model_cfg = self.model_cfg
        train_cfg = self.train_cfg
        data_reader = PoseDataReader(self.data_cfg)
        dataset = data_reader.read_data(train_cfg)

        _heatmpa_fn = functools.partial(
            keypoints_to_heatmaps_and_vectors,
            pairs=self.pairs,
            grid_shape=model_cfg.output_shape,
            window_size=train_cfg.window_size,
            vector_scale=train_cfg.vector_scale,
            offset_scale=train_cfg.offset_scale
        )

        def heatmap_fn(image, keypoints, bboxes, mask):
            heatmaps, vecmaps, offsetmaps, mask = tf.py_func(
                _heatmpa_fn, [keypoints, mask],
                [tf.float32, tf.float32, tf.float32, tf.float32])
            return image, heatmaps, vecmaps, offsetmaps, mask

        dataset = dataset.map(
            heatmap_fn,
            num_parallel_calls=train_cfg.num_parallel_map_calls
        )
        dataset = dataset.prefetch(train_cfg.prefetch_size)

        def map_fn(images, heatmaps, vecmaps, offsetmaps, masks):
            features = {'images': images}
            # masks.set_shape(model_cfg.input_shape)
            # masks = tf.expand_dims(masks, axis=-1)
            # masks = tf.image.resize_images(
            #     masks, size=model_cfg.output_shape)
            # masks = tf.squeeze(masks)
            labels = {'heatmaps': heatmaps,
                      'vecmaps': vecmaps,
                      'offsetmaps': offsetmaps,
                      'masks': masks}
            return features, labels

        dataset = dataset.map(
            map_fn, num_parallel_calls=train_cfg.num_parallel_map_calls)
        dataset = dataset.prefetch(train_cfg.prefetch_size)
        # if train_cfg.shuffle:
        #     dataset = dataset.shuffle(train_cfg.shuffle_buffer_size)
        dataset = dataset.repeat(train_cfg.num_epochs or None)
        dataset = dataset.batch(train_cfg.batch_size)
        dataset = dataset.prefetch(train_cfg.prefetch_size)
        return dataset

    def prepare_tf_summary(self, features, predictions, max_display=3):
        batch_size = self.train_cfg.batch_size
        images_in = tf.cast(features['images'], tf.uint8)
        images = tf.split(
            images_in,
            num_or_size_splits=batch_size,
            axis=0)
        heatmaps_logits = predictions['heatmaps']
        vecmaps = predictions['vecmaps'] * self.train_cfg.vector_scale
        offsetmaps = predictions['offsetmaps'] * self.train_cfg.offset_scale
        heatmaps = tf.nn.sigmoid(heatmaps_logits)
        heatmaps = non_max_suppression(heatmaps, self.train_cfg.window_size)

        heatmaps = tf.split(
            heatmaps,
            num_or_size_splits=batch_size,
            axis=0)
        vecmaps = tf.split(
            vecmaps,
            num_or_size_splits=batch_size,
            axis=0)
        offsetmaps = tf.split(
            offsetmaps,
            num_or_size_splits=batch_size,
            axis=0)
        heatmap_out = []

        heatmap_vis_fn = functools.partial(
            vis.visualize_heatmaps,
            pairs=self.pairs,
            threshold=0.2)

        for i in range(max_display):
            image_i = tf.squeeze(images[i])
            heatmaps_i = tf.squeeze(heatmaps[i])
            vecmaps_i = tf.squeeze(vecmaps[i])
            offsetmaps_i = tf.squeeze(offsetmaps[i])
            out = tf.py_func(
                heatmap_vis_fn,
                [image_i, heatmaps_i, vecmaps_i, offsetmaps_i],
                tf.uint8)
            heatmap_out.append(tf.expand_dims(out, axis=0))

        heatmap_out = tf.concat(heatmap_out, axis=0)
        tf.summary.image('images', images_in[:max_display], max_display)
        tf.summary.image('heatmap', heatmap_out, max_display)

    def train(self):
        """run training experiment"""
        if use_hvd:
            hvd.init()
            session_config = tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    allow_growth=True,
                    visible_device_list=str(hvd.local_rank())
                ))
        else:
            session_config = tf.ConfigProto(
                allow_soft_placement=True
            )

        if not os.path.exists(self.train_cfg.model_dir):
            os.makedirs(self.train_cfg.model_dir)

        model_path = os.path.join(
            self.train_cfg.model_dir,
            self.model_cfg.model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.hparams.model_dir = model_path

        model_dir = model_path
        if use_hvd and (hvd.rank() != 0):
            # Horovod: save checkpoints only on worker 0
            # to prevent other workers from corrupting them.
            model_dir = None

        run_config = tf.contrib.learn.RunConfig(
            model_dir=model_dir,
            session_config=session_config
        )

        estimator = tf.estimator.Estimator(
            model_fn=self.get_model_fn(),
            params=self.hparams,  # HParams
            config=run_config  # RunConfig
        )

        hooks = None
        if use_hvd:
            # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states from
            # rank 0 to all other processes. This is necessary to ensure consistent
            # initialization of all workers when training is started with random weights or
            # restored from a checkpoint.
            bcast_hook = hvd.BroadcastGlobalVariablesHook(0)
            hooks = [bcast_hook]

        def train_input_fn():
            """Create input graph for model.
            """
            # TODO : add multi-gpu training
            with tf.device(self.cpu_device):
                dataset = self.get_features_labels_data()
                return dataset

        # train_input_fn = self.input_fn
        estimator.train(input_fn=train_input_fn,
                        hooks=hooks)

    def get_optimizer_fn(self):
        """returns an optimizer function
        which takes as argument learning rate"""
        opt = dict(self.train_cfg.optimizer)
        opt_name = opt.pop('name', None)

        if opt_name == 'adam':
            opt_params = opt.pop('params', {})
            # remove learning rate if present
            opt_params.pop('learning_rate', None)

            def optimizer_fn(lr):
                opt = tf.train.AdamOptimizer(lr)
                if use_hvd:
                    return hvd.DistributedOptimizer(opt)
                else:
                    return opt

        else:
            raise NotImplementedError(
                "Optimizer {} not yet implemented".format(opt_name))

        return optimizer_fn

    def get_train_op(self, loss):
        """Get the training Op.
        Args:
             loss (Tensor): Scalar Tensor that represents the loss function.
        Returns:
            Training Op
        """
        # TODO: build configurable optimizer
        # optimizer_cfg = train_cfg.optimizer

        learning_rate = self.train_cfg.learning_rate
        if use_hvd:
            learning_rate *= hvd.size()
        lr_decay_params = self.train_cfg.learning_rate_decay
        if lr_decay_params is not None:
            lr_decay_fn = functools.partial(
                tf.train.exponential_decay,
                decay_steps=lr_decay_params['decay_steps'],
                decay_rate=lr_decay_params['decay_rate'],
                staircase=True
            )
        else:
            lr_decay_fn = None

        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=self.get_optimizer_fn(),
            learning_rate=learning_rate,
            learning_rate_decay_fn=lr_decay_fn
        )

    @staticmethod
    def get_eval_metric_ops(labels, predictions):
        """Return a dict of the evaluation Ops.
        Args:
            labels (Tensor): Labels tensor for training and evaluation.
            predictions (Tensor): Predictions Tensor.
        Returns:
            Dict of metric results keyed by name.
        """
        return {
            'Accuracy': tf.metrics.accuracy(
                labels=labels,
                predictions=predictions,
                name='accuracy')
        }

    def get_model_fn(self):
        """Return the model_fn.
        """
        # TODO: add multi-GPU training and CPU/GPU optimizations
        train_cfg = self.train_cfg

        def model_fn(features, labels, mode, params):
            """Model function used in the estimator.
            Args:
                model (Model): an instance of class Model
                features (Tensor): Input features to the model.
                labels (Tensor): Labels tensor for training and evaluation.
                mode (ModeKeys): Specifies if training, evaluation or prediction.
                params (HParams): hyperparameters.
            Returns:
                (EstimatorSpec): Model to be run by Estimator.
            """
            model = None
            model_name = params.model_name
            print("Using model ", model_name)
            if model_name == 'mobilenet_pose':
                model = MobilenetPose(params)
            else:
                NotImplementedError("{} not implemented".format(model_name))

            is_training = mode == tf.estimator.ModeKeys.TRAIN
            # Define model's architecture
            # inputs = {'images': features}
            # predictions = model.predict(inputs, is_training=is_training)
            predictions = model.predict(features, is_training=is_training)
            with tf.device(self.cpu_device):
                self.prepare_tf_summary(features, predictions)
            # Loss, training and eval operations are not needed during inference.
            loss = None
            train_op = None
            eval_metric_ops = {}
            if mode != tf.estimator.ModeKeys.PREDICT:
                # labels = tf.image.resize_bilinear(
                #     labels, size=params.output_shape)
                # heatmaps = labels[:, :, :, :-1]
                # masks = tf.squeeze(labels[:, :, :, -1])
                # labels = heatmaps
                # ground_truth = {'heatmaps': heatmaps,
                #                 'masks': masks}
                ground_truth = labels
                losses = model.losses(predictions, ground_truth)
                with tf.device(self.cpu_device):
                    for loss_name, loss_val in losses.items():
                        tf.summary.scalar('loss/' + loss_name, loss_val)
                # with tf.device(self.param_server_device):
                loss = losses['heatmap_loss']
                loss += train_cfg.vecmap_loss_weight * losses['vecmap_loss']
                loss += train_cfg.offsetmap_loss_weight * losses['offsetmap_loss']
                if self.train_cfg.quantize:
                    # Call the training rewrite which rewrites the graph in-place with
                    # FakeQuantization nodes and folds batchnorm for training. It is
                    # often needed to fine tune a floating point model for quantization
                    # with this training tool. When training from scratch, quant_delay
                    # can be used to activate quantization after training to converge
                    # with the float graph, effectively fine-tuning the model.
                    tf.contrib.quantize.create_training_graph(
                        tf.get_default_graph(), quant_delay=20000)
                train_op = self.get_train_op(loss)
                eval_metric_ops = None  # get_eval_metric_ops(labels, predictions)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops
            )

        return model_fn

    def freeze_model(self):
        # We retrieve our checkpoint fullpath
        checkpoint = tf.train.get_checkpoint_state(self.infer_cfg.model_dir)
        input_checkpoint = checkpoint.model_checkpoint_path

        # We precise the file fullname of our freezed graph

        absolute_model_dir = os.path.dirname(input_checkpoint)
        output_graph = os.path.join(absolute_model_dir, "frozen_model.pb")

        model, output_nodes = None, None
        model_name = self.hparams.model_name
        print("Using model ", model_name)
        if model_name == 'mobilenet_pose':
            model = MobilenetPose(self.hparams)
        else:
            NotImplementedError("{} not implemented".format(model_name))

        h, w = self.infer_cfg.network_input_shape
        inputs = {'images': tf.placeholder(tf.float32, [None, h, w, 3],
                                           name='images')}
        predictions = model.predict(inputs, is_training=False)
        if self.train_cfg.quantize:
            # Call the eval rewrite which rewrites the graph in-place with
            # FakeQuantization nodes and fold batchnorm for eval.
            tf.contrib.quantize.create_eval_graph()
        heatmaps = tf.nn.sigmoid(predictions['heatmaps'], name='heatmaps')
        vecmaps = tf.identity(predictions['vecmaps'], name='vecmaps')
        offsetmaps = tf.identity(predictions['offsetmaps'], name='offsetmaps')
        # bbox_classes = tf.nn.softmax(bbox_clf_logits, name='bbox_classes')
        # bbox_regs = tf.identity(predictions['bbox_regs'], name='bbox_regs')

        output_nodes = ['heatmaps', 'vecmaps', 'offsetmaps']

        for n in tf.get_default_graph().as_graph_def().node:
            print(n.name)

        if DEBUG:
            # TODO : load only required variables from checkpoint
            reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
            checkpoint_vars = reader.get_variable_to_shape_map()
            checkpoint_vars = [v for v in tf.trainable_variables()
                               if v.name.split(":")[0] in checkpoint_vars.keys()]
            saver = tf.train.Saver(checkpoint_vars)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, input_checkpoint)

            # We use a built-in TF helper to export variables to constants
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,  # The session is used to retrieve the weights
                tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
                output_nodes  # The output node names are used to select the useful nodes
            )

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile(output_graph, "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))
            # tf.train.write_graph(output_graph_def, absolute_model_dir,
            #                      "frozen_model.pbtxt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str,
                        default='./config.yaml', help='Config file')
    args = parser.parse_args()
    config_file = args.config_file
    assert os.path.exists(config_file), \
        "{} not found".format(config_file)
    trainer = Trainer(config_file)
    # trainer.train()
    trainer.freeze_model()
