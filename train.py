import os
import argparse
import tensorflow as tf
from model.mobilenet_pose import MobilenetPose
import functools
from dataset.data_reader import PoseDataReader
from utils.parse_config import parse_config
from utils.bboxes import (generate_anchors, get_matches,
                          bbox_decode)
from utils.dataset_util import keypoints_to_heatmap
from utils.ops import non_max_suppression
import utils.visualize as vis

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

    def generate_anchors(self):
        all_anchors = []
        for i, (base_anchor_size, stride) in enumerate(zip(
                self.model_cfg.base_anchor_sizes,
                self.model_cfg.anchor_strides)):
            grid_shape = tf.constant(
                self.model_cfg.input_shape, tf.int32) / stride
            anchors = generate_anchors(
                grid_shape=grid_shape,
                base_anchor_size=base_anchor_size,
                stride=stride,
                scales=self.model_cfg.anchor_scales,
                aspect_ratios=self.model_cfg.anchor_ratios)
            all_anchors.append(anchors)
        return tf.concat(all_anchors, axis=0)

    def get_features_labels_data(self):
        """returns dataset containing (features, labels)"""
        model_cfg = self.model_cfg
        train_cfg = self.train_cfg
        anchors = self.generate_anchors()
        num_keypoints = len(train_cfg.train_keypoints)
        data_reader = PoseDataReader(self.data_cfg)
        dataset = data_reader.read_data(train_cfg)
        heatmap_fn = functools.partial(
            keypoints_to_heatmap,
            num_keypoints=num_keypoints,
            grid_shape=model_cfg.output_shape)
        dataset = dataset.map(
            heatmap_fn,
            num_parallel_calls=train_cfg.num_parallel_map_calls
        )
        dataset = dataset.prefetch(train_cfg.prefetch_size)

        def map_fn(images, heatmaps, bboxes, masks):
            features = {'images': images}
            classes, regs, weights = get_matches(
                gt_bboxes=bboxes,
                pred_bboxes=anchors,
                unmatched_threshold=model_cfg.unmatched_threshold,
                matched_threshold=model_cfg.matched_threshold,
                force_match_for_gt_bbox=model_cfg.force_match_for_gt_bbox,
                scale_factors=model_cfg.scale_factors)
            bbox_labels = {'classes': classes,
                           'regs': regs,
                           'weights': weights}
            masks.set_shape(model_cfg.input_shape)
            masks = tf.expand_dims(masks, axis=-1)
            masks = tf.image.resize_images(
                masks, size=model_cfg.output_shape)
            masks = tf.squeeze(masks)
            labels = {'heatmaps': heatmaps,
                      'masks': masks,
                      'bboxes': bbox_labels}
            return features, labels

        dataset = dataset.map(
            map_fn, num_parallel_calls=train_cfg.num_parallel_map_calls)
        dataset = dataset.prefetch(train_cfg.prefetch_size)
        dataset = dataset.batch(train_cfg.batch_size)
        dataset = dataset.prefetch(train_cfg.prefetch_size)
        return dataset

    def prepare_tf_summary(self, features, predictions, max_display=3):
        all_anchors = self.generate_anchors()
        batch_size = self.train_cfg.batch_size
        images = tf.cast(features['images'], tf.uint8)
        images = tf.split(
            images,
            num_or_size_splits=batch_size,
            axis=0)
        heatmaps_logits = predictions['heatmaps']
        heatmaps = tf.nn.sigmoid(heatmaps_logits)
        heatmaps = non_max_suppression(heatmaps, 3)
        # heatmap_out = tf.expand_dims(
        #     tf.zeros_like(heatmaps[:, :, :, 0]), axis=-1)
        # heatmap_out = tf.concat([heatmap_out, heatmaps], -1)
        heatmaps = tf.split(
            heatmaps,
            num_or_size_splits=batch_size,
            axis=0)
        heatmap_out = []
        for i in range(max_display):
            image_i = tf.squeeze(images[i])
            heatmaps_i = tf.squeeze(heatmaps[i])
            out = tf.py_func(
                vis.visualize_heatmaps, [image_i, heatmaps_i, 0.2],
                tf.uint8)
            heatmap_out.append(tf.expand_dims(out, axis=0))
        bbox_clf_logits = predictions['bbox_clf_logits']
        _, bbox_probs = tf.split(
            tf.nn.softmax(bbox_clf_logits),
            num_or_size_splits=2,
            axis=1)
        bbox_probs = tf.split(
            tf.squeeze(bbox_probs),
            num_or_size_splits=batch_size,
            axis=0)
        bbox_regs = tf.split(
            predictions['bbox_regs'],
            num_or_size_splits=batch_size,
            axis=0)
        out_images = []

        for i in range(max_display):
            indices = tf.squeeze(tf.where(
                tf.greater(bbox_probs[i], 0.5)))

            def _draw_bboxes():
                img = tf.squeeze(images[i])
                bboxes = tf.gather(bbox_regs[i], indices)
                # bboxes = tf.zeros_like(bboxes)
                anchors = tf.gather(all_anchors, indices)
                bboxes = bbox_decode(
                    bboxes, anchors, self.model_cfg.scale_factors)
                # bboxes = tf.expand_dims(bboxes, axis=0)
                scores = tf.gather(bbox_probs[i], indices)
                selected_indices = tf.image.non_max_suppression(
                    bboxes, scores,
                    max_output_size=10,
                    iou_threshold=0.5)
                bboxes = tf.gather(bboxes, selected_indices)
                out_img = tf.py_func(vis.visualize_bboxes_on_image,
                                     [img, bboxes], tf.uint8)
                return tf.expand_dims(out_img, axis=0)
                # return tf.image.draw_bounding_boxes(
                #    images[i], bboxes)

            out_image = tf.cond(
                tf.greater(tf.rank(indices), 0),
                true_fn=_draw_bboxes,
                false_fn=lambda: images[i])
            out_images.append(out_image)

        out_images = tf.concat(out_images, axis=0)
        heatmap_out = tf.concat(heatmap_out, axis=0)
        tf.summary.image('bboxes', out_images, max_display)
        tf.summary.image('heatmap', heatmap_out, max_display)

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

        def train_input_fn():
            """Create input graph for model.
            """
            # TODO : add multi-gpu training
            with tf.device('/cpu:0'):
                dataset = self.get_features_labels_data()
                return dataset

        # train_input_fn = self.input_fn
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
                for loss_name, loss_val in losses.items():
                    tf.summary.scalar('loss/' + loss_name, loss_val)
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

    # @staticmethod
    # def input_fn():
    #     """Create input graph for model.
    #     """
    #     # TODO : add multi-gpu training
    #     with tf.device('/cpu:0'):
    #         dataset = Trainer.get_features_labels_data()
    #         return dataset


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
