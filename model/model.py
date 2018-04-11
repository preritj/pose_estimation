"""
Base class for pose estimation
"""

import tensorflow as tf


class Model:
    def __init__(self):
        self._is_training = True
        pass

    def preprocess(self, inputs):
        """Image preprocessing"""
        pass

    def encoder(self, preprocessed_inputs):
        """Generates features"""
        pass

    def decoder(self, image_features):
        pass


