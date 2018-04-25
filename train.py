import os
import argparse
import tensorflow as tf
from model.mobilenet_pose import MobilenetPose
import functools
from dataset.data_reader import PoseDataReader
from utils.parse_config import parse_config

slim = tf.contrib.slim


def get_optimizer_fn(optimizer_cfg):
    """Takes as input optimizer config
    e.g. {'name': 'adam',
          'params': {'beta1': 0.9, 'beta2': 0.99}}
    returns an optimizer function which takes as argument learning rate"""
    opt = dict(optimizer_cfg)
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


def get_train_op(loss, params):
    """Get the training Op.
    Args:
         loss (Tensor): Scalar Tensor that represents the loss function.
         params (HParams): Hyperparameters
           (needs to have `learning_rate` and 'optimizer')
    Returns:
        Training Op
    """
    optimizer_cfg = params.optimizer

    lr_decay_params = params.learning_rate_decay
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
        optimizer=get_optimizer_fn(optimizer_cfg),
        learning_rate=params.learning_rate,
        learning_rate_decay_fn=lr_decay_fn
    )


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


def get_model_fn(model_cfg):
    """Return the model_fn.
    Args:
        model_cfg :
        run_config (RunConfig): Configuration for Estimator run.
    """
    # TODO: add multi-GPU training and CPU/GPU optimizations

    model_name = model_cfg.model_name
    print("Using model ", model_name)
    if model_name == 'mobilenet_pose':
        model = MobilenetPose(model_cfg)
    else:
        NotImplementedError("{} not implemented".format(model_name))

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
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        # Define model's architecture
        inputs = {'images': features}
        predictions_dict = model.predict(inputs, is_training=is_training)
        predictions = predictions_dict['heatmaps']
        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}
        labels = tf.image.resize_bilinear(
            labels, size=model_cfg.output_shape)
        heatmaps = labels[:, :, :, :-1]
        masks = tf.squeeze(labels[:, :, :, -1])
        labels = heatmaps
        if mode != tf.estimator.ModeKeys.PREDICT:
            ground_truth = {'heatmaps': heatmaps,
                            'masks': masks}
            loss = model.losses(predictions_dict, ground_truth)
            loss = loss['l2_loss']
            train_op = get_train_op(loss, params)
            eval_metric_ops = get_eval_metric_ops(labels, predictions)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )

    return model_fn


def input_fn(data_reader, train_cfg):
    """Create input graph for model.
    Args:
      data_reader: PoseDataReader instance
      train_cfg: training parameters
    Returns:
      features, labels
    """
    # TODO : add multi-gpu training
    with tf.device('/cpu:0'):
        dataset = data_reader.get_features_labels_data(train_cfg)
        return dataset


# Define and run experiment
def run_experiment(config_file):
    """Run the training experiment."""
    # Define model parameters
    cfg = parse_config(config_file)
    data_cfg = cfg['data_config']
    train_cfg = cfg['train_config']
    model_cfg = cfg['model_config']

    hparams = tf.contrib.training.HParams(
        **data_cfg.__dict__,
        **train_cfg.__dict__,
        **model_cfg.__dict__,
        num_keypoints=len(train_cfg.train_keypoints))

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            force_gpu_compatible=True,
            allow_growth=True)
    )

    if not os.path.exists(hparams.model_dir):
        os.makedirs(train_cfg.model_dir)

    model_path = os.path.join(
        train_cfg.model_dir, model_cfg.model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    run_config = tf.contrib.learn.RunConfig(
        model_dir=train_cfg.model_dir,
        session_config=session_config
    )

    estimator = tf.estimator.Estimator(
        model_fn=get_model_fn(hparams),
        params=hparams,  # HParams
        config=run_config  # RunConfig
    )

    data_reader = PoseDataReader(data_cfg)
    train_input_fn = functools.partial(
        input_fn,
        data_reader=data_reader,
        train_cfg=train_cfg)

    estimator.train(input_fn=train_input_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str,
                        default='./config.yaml', help='Config file')
    args = parser.parse_args()
    config_file = args.config_file
    assert os.path.exists(config_file), \
        "{} not found".format(config_file)
    run_experiment(config_file)
