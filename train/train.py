import tensorflow as tf
from model.base_model import Model

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


def get_configs(train_cfg):
    hparams = tf.contrib.training.HParams(
        train_steps=args.ep * cifar10.Cifar10DataSet.num_examples_per_epoch() // args.bsize,
        iters_ep=cifar10.Cifar10DataSet.num_examples_per_epoch() // args.bsize,
        n_classes=10,
        **train_cfg
    )
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=args.dev_place,
        gpu_options=tf.GPUOptions(
            force_gpu_compatible=True,
            allow_growth=True)
    )
    run_config = tf.contrib.learn.RunConfig(
        model_dir=train_cfg.model_dir,
        tf_random_seed=args.rseed,
        save_checkpoints_steps=hparams.iters_ep,
        log_step_count_steps=hparams.iters_ep, # only log every epoch
        session_config=session_config
    )
    return hparams, run_config


def get_model_fn(model, train_cfg):
    assert isinstance(model, Model), "model type not supported"

    def model_fn(features, labels, mode, params):
        """Model function used in the estimator.
        Args:
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
        prediction = model.predict(inputs, is_training=is_training)
        heatmaps = prediction['heatmaps']
        # Loss, training and eval operations are not needed during inference.
        loss = None
        train_op = None
        eval_metric_ops = {}
        if mode != tf.estimator.ModeKeys.PREDICT:
            loss = model.losses(heatmaps, labels)
            optimizer = _build_optimizer(train_cfg)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)
            eval_metric_ops = get_eval_metric_ops(labels, predictions)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=heatmaps,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops
        )




# def build_optimizer(train_cfg):
#     opt = dict(train_cfg.optimizer)
#     opt_name = opt.pop('name', None)
#     if opt_name == 'adam':
#         opt_params = opt.pop('params', {})
#         opt_params['learning_rate'] = train_cfg.learning_rate
#         optimizer = tf.train.AdamOptimizer(**opt)
#     else:
#         raise NotImplementedError(
#             "Optimizer {} not yet implemented".format(opt_name))
#     return optimizer
#
#
# def train(train_cfg, data_reader):
#     with tf.Graph().as_default():
#         # TODO: Build a configuration specifying multi-GPU and multi-replicas.
#         # for now we use default config i.e. lonely worker
#         deploy_config = model_deploy.DeploymentConfig()
#
#         # Create the global step on the device storing the variables.
#         with tf.device(deploy_config.variables_device()):
#             global_step = slim.create_global_step()
#
#         with tf.device(deploy_config.inputs_device()):
#             dataset = data_reader.read_data(train_cfg)
#             iterator = dataset.make_initializable_iterator()
#
#         # Define the optimizer.
#         with tf.device(deploy_config.optimizer_device()):
#             optimizer = build_optimizer(train_cfg)
#             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#             with tf.control_dependencies(update_ops):
#                 train_op = optimizer.minimize(loss, global_step=global_step)
#
#         def _initializer_fn(sess):
#             iterator.initializer



