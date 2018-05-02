import os
import argparse
import tensorflow as tf
from model.mobilenet_pose import MobilenetPose
import functools
from dataset.data_reader import PoseDataReader
from utils.parse_config import parse_config

slim = tf.contrib.slim


class Trainer(object):

    def __init__(self, cfg_file):
        # Define model parameters
        cfg = parse_config(cfg_file)
        self.data_cfg = cfg['data_config']
        self.train_cfg = cfg['train_config']
        self.model_cfg = cfg['model_config']
        self.hparams = tf.contrib.training.HParams(
            **self.model_cfg.__dict__,
            num_keypoints=len(self.train_cfg.train_keypoints))
        self.data_reader = PoseDataReader(self.data_cfg)

    def train(self):
        """run training experiment"""
        session_config = tf.ConfigProto(
            allow_soft_placement=True  # ,
            # log_device_placement=False,
            # gpu_options=tf.GPUOptions(
            #     force_gpu_compatible=True,
            #     allow_growth=True)
        )

        if not os.path.exists(self.train_cfg.model_dir):
            os.makedirs(self.train_cfg.model_dir)

        model_path = os.path.join(
            self.train_cfg.model_dir,
            self.model_cfg.model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.hparams.model_dir = model_path

        run_config = tf.contrib.learn.RunConfig(
            model_dir=self.train_cfg.model_dir,
            session_config=session_config
        )

        estimator = tf.estimator.Estimator(
            model_fn=self.get_model_fn(),
            params=self.hparams,  # HParams
            config=run_config  # RunConfig
        )

        train_input_fn = self.input_fn
        estimator.train(input_fn=train_input_fn)

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
                return tf.train.AdamOptimizer(lr, **opt_params)

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
            optimizer=tf.train.AdamOptimizer,  # get_optimizer_fn(optimizer_cfg),
            learning_rate=self.train_cfg.learning_rate,
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
            heatmaps_logits = predictions['heatmaps']
            heatmaps = tf.nn.sigmoid(heatmaps_logits)
            summary_out = tf.expand_dims(
                tf.zeros_like(heatmaps[:, :, :, 0]), axis=-1)
            summary_out = tf.concat([summary_out, heatmaps], -1)
            tf.summary.image('image', features['images'], max_outputs=3)
            tf.summary.image('heatmap', summary_out, max_outputs=3)
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
                for loss_name, loss_val in losses.items():
                    tf.summary.scalar(loss_name, loss_val)
                loss = losses['heatmap_loss']
                loss += train_cfg.bbox_clf_weight * losses['bbox_clf_loss']
                loss += train_cfg.bbox_reg_weight * losses['bbox_reg_loss']
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

    def input_fn(self):
        """Create input graph for model.
        """
        # TODO : add multi-gpu training
        with tf.device('/cpu:0'):
            dataset = self.data_reader.get_features_labels_data(
                self.train_cfg, self.model_cfg)
            return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str,
                        default='./config.yaml', help='Config file')
    args = parser.parse_args()
    config_file = args.config_file
    assert os.path.exists(config_file), \
        "{} not found".format(config_file)
    trainer = Trainer(config_file)
    trainer.train()
