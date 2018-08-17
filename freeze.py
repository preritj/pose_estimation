import os
import argparse
import tensorflow as tf
from model.mobilenet_pose import MobilenetPose
from train import Trainer
from tensorflow.python import pywrap_tensorflow

DEBUG = False


class FreezeModel(Trainer):
    def __init__(self, cfg_file):
        super().__init__(cfg_file)

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
        inputs = {'images': tf.placeholder(tf.float32, [None, None, None, 3],
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
    freezer = FreezeModel(config_file)
    freezer.freeze_model()